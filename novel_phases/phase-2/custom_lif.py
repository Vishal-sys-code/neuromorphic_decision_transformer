import torch
import torch.nn as nn
from norse.torch import LIFCell
from norse.torch.functional.lif import LIFParameters
from typing import Optional, Tuple, NamedTuple


class CustomLIFCellState(NamedTuple):
    v: torch.Tensor
    i: torch.Tensor
    eligibility_trace: torch.Tensor


class CustomLIFCell(LIFCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIFParameters = LIFParameters(),
        dt: float = 0.001,
        trace_decay: float = 0.95,
        name: Optional[str] = None, # Added for compatibility with Norse's base cell
        **kwargs
    ):
        super().__init__(p=p, dt=dt, name=name, **kwargs) # Pass name and kwargs to parent
        self.input_size = input_size
        self.hidden_size = hidden_size # Corresponds to output_size for a single layer
        self.trace_decay = trace_decay

        # Initialize eligibility trace
        # This trace is typically associated with the weights of a layer.
        # If this cell IS the layer, trace shape is (input_size, hidden_size)
        # If this cell is PART of a more complex layer (e.g. recurrent), this might differ.
        # For now, assuming it's for a feed-forward connection where this cell's output
        # is the post-synaptic activity and its input is the pre-synaptic activity
        # for weights connecting input_size to hidden_size.
        self.register_buffer(
            "eligibility_trace",
            torch.zeros(input_size, hidden_size, device=kwargs.get('device'), dtype=kwargs.get('dtype'))
        )

    def get_initial_state(self, batch_size: int, inputs: Optional[torch.Tensor] = None) -> CustomLIFCellState:
        # Overriding to include eligibility trace in the state if needed,
        # but the trace is more of a persistent parameter of the cell for learning,
        # rather than a state that changes with each input in a sequence in the same way v and i do.
        # For now, the eligibility trace is stored directly in the module.
        # If we need per-sequence traces, this would need to change.
        s_prev = super().get_initial_state(batch_size, inputs)
        # The eligibility trace is not part of the recurrent state passed from step to step.
        # It's a module buffer that accumulates over time.
        # So, we return the parent's state directly.
        return s_prev # v, i

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs a forward pass through the CustomLIFCell.

        Args:
            x (torch.Tensor): Input tensor (typically spikes) of shape (batch_size, input_size).
            state (Optional[Tuple[torch.Tensor, torch.Tensor]]): Previous state (v, i).
                                                               If None, it's initialized.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - Output spikes (s_out) of shape (batch_size, hidden_size).
                - Next state (v_next, i_next).
        """
        if state is None:
            # If this cell is the first in a sequence or state is not passed,
            # it uses its own internal state, which is fine for non-recurrent use.
            # For recurrent use, state should be explicitly managed.
            # Norse LIFCell's default behavior handles this if state is None.
            # We get the initial state for v and i from the parent.
            # The eligibility trace is handled separately as it persists across calls differently.
            initial_parent_state = super().get_initial_state(batch_size=x.shape[0], inputs=x)
            if state is None:
                state = initial_parent_state


        # Perform the standard LIF cell computation
        s_out, next_state = super().forward(x, state) # next_state is (v_next, i_next)

        # Update eligibility trace
        # x contains pre-synaptic spikes (batch_size, input_size)
        # s_out contains post-synaptic spikes (batch_size, hidden_size)
        # We need to compute the outer product for each item in the batch and sum or average.

        # Assuming x is pre-synaptic spikes (0 or 1) and s_out is post-synaptic spikes (0 or 1)
        # For a batch, we sum the outer products: sum_batch(pre_i.T @ post_j)
        if x.requires_grad: # Ensure pre_spikes are detached if they come from a part of the graph we don't want to influence via this path
            pre_spikes = x.detach()
        else:
            pre_spikes = x

        if s_out.requires_grad:
            post_spikes = s_out.detach()
        else:
            post_spikes = s_out

        # Sum over the batch dimension
        # pre_spikes: (batch_size, input_size)
        # post_spikes: (batch_size, hidden_size)
        # update should be (input_size, hidden_size)
        # (input_size, batch_size) @ (batch_size, hidden_size)
        batch_trace_update = torch.matmul(pre_spikes.t(), post_spikes) / x.shape[0] # Averaging over batch

        self.eligibility_trace.mul_(self.trace_decay).add_(batch_trace_update)

        # The state returned should match what the parent LIFCell returns for recurrent connections.
        # The eligibility trace is updated in-place within the module.
        return s_out, next_state

    def reset_trace(self):
        """Resets the eligibility trace to zeros."""
        self.eligibility_trace.zero_()

# Example Usage (Illustrative)
if __name__ == '__main__':
    batch_size = 10
    input_features = 20
    output_features = 5 # hidden_size for the cell

    # Create a CustomLIFCell
    custom_lif_cell = CustomLIFCell(input_features, output_features)

    # Dummy input spikes (binary) and initial state
    # Typically, input spikes would be generated by a previous layer or Poisson encoder
    input_spikes = (torch.rand(batch_size, input_features) > 0.8).float()

    # Get initial state for v and i from the cell itself
    # This is how Norse typically handles it if you don't pass a state.
    # The state is managed internally by the cell if not provided.
    initial_state = custom_lif_cell.get_initial_state(batch_size=batch_size, inputs=input_spikes)


    # Simulate a few time steps
    print(f"Initial eligibility trace:\n{custom_lif_cell.eligibility_trace}")

    # First step
    print("\n--- Step 1 ---")
    s_out, next_state = custom_lif_cell(input_spikes, initial_state)
    print(f"Output spikes (shape: {s_out.shape}):\n{s_out}")
    print(f"Updated eligibility trace (shape: {custom_lif_cell.eligibility_trace.shape}):\n{custom_lif_cell.eligibility_trace}")

    # Second step (using the state from the previous step)
    print("\n--- Step 2 ---")
    input_spikes_2 = (torch.rand(batch_size, input_features) > 0.7).float()
    s_out_2, next_state_2 = custom_lif_cell(input_spikes_2, next_state)
    print(f"Output spikes 2 (shape: {s_out_2.shape}):\n{s_out_2}")
    print(f"Updated eligibility trace:\n{custom_lif_cell.eligibility_trace}")

    # Reset trace
    custom_lif_cell.reset_trace()
    print(f"\nAfter reset, eligibility trace:\n{custom_lif_cell.eligibility_trace}")

    # Test with a different device if available
    if torch.cuda.is_available():
        print("\n--- CUDA Test ---")
        device = torch.device("cuda")
        custom_lif_cell_cuda = CustomLIFCell(input_features, output_features, device=device, dtype=torch.float32)
        input_spikes_cuda = input_spikes.to(device)
        initial_state_cuda = custom_lif_cell_cuda.get_initial_state(batch_size=batch_size, inputs=input_spikes_cuda)

        s_out_cuda, _ = custom_lif_cell_cuda(input_spikes_cuda, initial_state_cuda)
        print(f"CUDA Output spikes (shape: {s_out_cuda.shape}) on device: {s_out_cuda.device}")
        print(f"CUDA Eligibility trace (shape: {custom_lif_cell_cuda.eligibility_trace.shape}) on device: {custom_lif_cell_cuda.eligibility_trace.device}:\n{custom_lif_cell_cuda.eligibility_trace}")

    print("\nNote: The eligibility trace accumulates. It's typically used in conjunction with a learning rule that applies it (and potentially resets it) after a learning episode/batch.")
    print("The CustomLIFCell itself doesn't return the eligibility trace in its forward pass's state tuple, as it's a module parameter.")
    print("If using this cell in a nn.Sequential or Norse's SequentialState, the state passed around will be (v,i).")
    print("The eligibility trace must be accessed directly from the module instance.")
