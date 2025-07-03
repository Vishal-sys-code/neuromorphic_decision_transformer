import torch
import torch.nn as nn
from typing import Optional

def apply_three_factor_update(
    layer_to_update: nn.Module,
    eligibility_trace: torch.Tensor,
    return_to_go: torch.Tensor, # Should be a scalar or broadcastable tensor
    local_learning_rate: float,
    clip_value: Optional[float] = None,
    normalize: bool = False
):
    """
    Applies a three-factor Hebbian-like update to the weights of a given layer.

    Args:
        layer_to_update (nn.Module): The layer whose weights are to be updated.
                                     Typically nn.Linear or nn.Conv2d.
                                     Must have a `weight` attribute.
        eligibility_trace (torch.Tensor): The eligibility trace accumulated for the weights.
                                          Shape should match layer_to_update.weight.data.
        return_to_go (torch.Tensor): The third factor, typically a scalar reward signal
                                     or return-to-go value. If it's a tensor, it should
                                     be broadcastable with the eligibility_trace.
        local_learning_rate (float): The learning rate for this local update rule.
        clip_value (Optional[float]): If provided, the absolute value of the computed
                                      weight change (delta_W) will be clipped to this value.
        normalize (bool): If True, the computed weight change (delta_W) will be normalized
                          (e.g., by its L2 norm) before being applied. This can help
                          stabilize updates. Clipping and normalization are mutually exclusive
                          in this implementation (normalization takes precedence).
    """
    if not hasattr(layer_to_update, 'weight'):
        raise ValueError("Layer to update must have a 'weight' attribute.")

    if layer_to_update.weight.data.shape != eligibility_trace.shape:
        raise ValueError(
            f"Shape mismatch: layer weights {layer_to_update.weight.data.shape}, "
            f"eligibility trace {eligibility_trace.shape}."
        )

    # Ensure return_to_go is on the same device as the trace and weights
    return_to_go = return_to_go.to(eligibility_trace.device)
    
    # Compute the raw weight update
    # ΔW_O = η_local * eligibility_trace * G_t
    # G_t can be a scalar or a tensor that needs broadcasting.
    # If G_t is a scalar, simple multiplication works.
    # If G_t needs to modulate specific parts of the trace (e.g. per output neuron),
    # its shape must be compatible. For now, assume G_t is a scalar.
    delta_W = local_learning_rate * eligibility_trace * return_to_go

    # Stability: Normalization or Clipping
    if normalize:
        norm = torch.norm(delta_W.detach(), p=2) # Using detach() to not affect gradient flow through delta_W if any
        if norm > 1e-9: # Avoid division by zero or very small norms
            delta_W = delta_W / norm
    elif clip_value is not None:
        delta_W = torch.clamp(delta_W, -clip_value, clip_value)

    # Apply the update to the layer's weights
    # This is done outside of PyTorch's autograd mechanism for the main backprop pass
    with torch.no_grad():
        layer_to_update.weight.data += delta_W

    # It's good practice to reset the eligibility trace after the update is applied,
    # assuming the trace is for a single "episode" or "batch" contributing to G_t.
    # The caller should handle this based on their specific logic for trace accumulation.
    # For example: custom_lif_cell_instance.reset_trace()
    
    return delta_W # Return the computed weight change for analysis

# Example Usage (Illustrative)
if __name__ == '__main__':
    from custom_lif import CustomLIFCell # Assuming custom_lif.py is in the same directory or PYTHONPATH

    # Setup
    input_dim = 10
    output_dim = 5
    batch_size = 1 # For simplicity in trace calculation for this example

    # Create a linear layer whose weights we want to update
    linear_layer = nn.Linear(input_dim, output_dim)
    print(f"Initial weights:\n{linear_layer.weight.data}")

    # Create a CustomLIFCell that would generate post-synaptic spikes for this linear layer
    # The eligibility trace in CustomLIFCell is (input_dim, output_dim)
    # This matches linear_layer.weight if linear_layer is considered the "current" layer
    # and input_dim is from the "previous" layer.
    lif_cell_for_trace = CustomLIFCell(input_dim, output_dim)

    # Simulate some pre and post synaptic activity to populate the trace
    # In a real scenario, this trace would build up over time steps in a sequence
    pre_spikes = (torch.rand(batch_size, input_dim) > 0.5).float() # (batch, input_dim)
    
    # To get post-synaptic spikes, we'd typically pass current through the linear layer, then to LIF
    # For this example, let's assume `lif_cell_for_trace` directly gives post-synaptic spikes
    # and its internal trace corresponds to `linear_layer`.
    # This part is a bit conceptual for a standalone example.
    # Let's manually set the trace for demonstration.
    # lif_cell_for_trace.eligibility_trace = torch.rand_like(linear_layer.weight.data) * 0.1
    
    # More realistically, let's simulate one step of the custom LIF cell
    # to get some trace values.
    # The `forward` of CustomLIFCell takes `x` which are pre-synaptic spikes to *itself*.
    # If `CustomLIFCell` *is* the layer, its input `x` is pre-synaptic.
    # If `CustomLIFCell` is *after* `linear_layer`, then `x` for `CustomLIFCell` is output of `linear_layer`.
    # Let's assume the trace in `lif_cell_for_trace` is the one we want to use.
    
    # Simulate a forward pass to populate the trace
    # This `input_current_for_lif` would be `linear_layer(pre_spikes)` if we model it fully.
    # For simplicity, let's assume `pre_spikes` are direct inputs to `lif_cell_for_trace`
    # and its trace is relevant for `linear_layer`.
    
    _ = lif_cell_for_trace(pre_spikes, None) # This updates lif_cell_for_trace.eligibility_trace
    current_trace = lif_cell_for_trace.eligibility_trace.clone()
    print(f"\nEligibility trace (simulated):\n{current_trace}")


    # Define other parameters for the update
    G_t = torch.tensor(1.5) # Example return-to-go (scalar)
    lr_local = 0.01
    clip = 0.1

    # Apply the update
    apply_three_factor_update(
        layer_to_update=linear_layer,
        eligibility_trace=current_trace,
        return_to_go=G_t,
        local_learning_rate=lr_local,
        clip_value=clip
    )
    print(f"\nWeights after update (with clipping):\n{linear_layer.weight.data}")

    # Reset for next test: Re-initialize weights and trace
    initial_weights = linear_layer.weight.data.clone()
    linear_layer.weight.data = initial_weights.clone() # Reset weights
    lif_cell_for_trace.reset_trace()
    _ = lif_cell_for_trace(pre_spikes, None) # Re-populate trace
    current_trace_norm = lif_cell_for_trace.eligibility_trace.clone()


    # Apply update with normalization
    apply_three_factor_update(
        layer_to_update=linear_layer,
        eligibility_trace=current_trace_norm,
        return_to_go=G_t,
        local_learning_rate=lr_local, # lr_local might need to be larger for normalized updates
        normalize=True
    )
    print(f"\nWeights after update (with normalization):\n{linear_layer.weight.data}")

    print("\nNote: The eligibility trace should ideally be reset by the training loop after an update.")
    lif_cell_for_trace.reset_trace() # Example of resetting
    print(f"Trace after reset: {torch.all(lif_cell_for_trace.eligibility_trace == 0)}")

    # Test with a different G_t (e.g. per output neuron)
    # This requires G_t to be broadcastable with eligibility_trace (input_dim, output_dim)
    # A G_t of shape (output_dim,) would work if it modulates based on output neuron activity/reward.
    print("\n--- Test with per-output G_t ---")
    linear_layer.weight.data = initial_weights.clone() # Reset weights
    lif_cell_for_trace.reset_trace()
    _ = lif_cell_for_trace(pre_spikes, None) # Re-populate trace
    current_trace_broadcast = lif_cell_for_trace.eligibility_trace.clone()

    G_t_vector = torch.randn(output_dim) # Example: one G_t value per output feature
    print(f"G_t_vector (shape {G_t_vector.shape}):\n{G_t_vector}")
    
    apply_three_factor_update(
        layer_to_update=linear_layer,
        eligibility_trace=current_trace_broadcast,
        return_to_go=G_t_vector.unsqueeze(0), # Make it (1, output_dim) for broadcasting with (input_dim, output_dim)
        local_learning_rate=lr_local,
        clip_value=clip
    )
    print(f"\nWeights after update (with G_t_vector):\n{linear_layer.weight.data}")

    # Example of error handling
    print("\n--- Test error handling ---")
    faulty_layer = nn.BatchNorm1d(output_dim) # Does not have 'weight' in the way Linear does for this rule
    try:
        apply_three_factor_update(faulty_layer, current_trace, G_t, lr_local)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    wrong_shape_trace = torch.randn(input_dim, output_dim + 1)
    try:
        apply_three_factor_update(linear_layer, wrong_shape_trace, G_t, lr_local)
    except ValueError as e:
        print(f"Caught expected error: {e}")

from typing import Optional # Added for clip_value type hint