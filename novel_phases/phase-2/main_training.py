import torch
import torch.nn as nn
import torch.optim as optim
from norse.torch import LIFState, PoissonEncoder # Using LIFState from norse.torch
from custom_lif import CustomLIFCell # Assuming this is in the same directory
from three_factor_updater import apply_three_factor_update # Assuming this is in the same directory

# For reproducibility
torch.manual_seed(0)

# Define a simple SNN model
class SimpleSNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_custom_lif=True):
        super(SimpleSNN, self).__init__()
        self.use_custom_lif = use_custom_lif
        self.encoder = PoissonEncoder(seq_length=10) # Encode input for 10 time steps

        # Layer 1: Linear layer followed by CustomLIFCell or standard LIFCell
        self.fc1 = nn.Linear(input_size, hidden_size)
        if self.use_custom_lif:
            # The CustomLIFCell will store the eligibility trace relevant to fc1's weights
            self.lif1 = CustomLIFCell(input_size=hidden_size, hidden_size=hidden_size, dt=0.001)
        else:
            # Using a standard LIFCell from Norse for comparison
            from norse.torch import LIFCell as StandardLIFCell # For comparison
            self.lif1 = StandardLIFCell(p=None, dt=0.001) # p=None uses default parameters


        # Layer 2: Output layer (e.g., another LIF layer or a simple linear readout)
        # For simplicity, let's use a linear readout layer for the final output
        # In a spiking context, this might be integrated spikes or another LIF layer.
        self.fc2 = nn.Linear(hidden_size, output_size)
        # To make it spiking, one might add another LIF after fc2, but let's keep it simple.

        self.output_activity_sum = [] # To store output layer activity for loss calculation

    def forward(self, x_static): # x_static is (batch_size, input_features)
        # Encode static input to spikes over time
        spikes_over_time = self.encoder(x_static) # (time_steps, batch_size, input_features)
        
        # Initialize states
        # For CustomLIFCell, state is (v,i). Trace is managed internally.
        # For StandardLIFCell, state is also (v,i).
        s1 = None # self.lif1.get_initial_state(x_static.shape[0], x_static.device) -> Norse cells handle this internally if None
        
        # List to store spikes from lif1 for fc2 input
        lif1_output_spikes_over_time = []

        for t in range(spikes_over_time.shape[0]): # Iterate over time steps
            x_t = spikes_over_time[t] # (batch_size, input_features)
            
            # Apply fc1. Note: For CustomLIFCell, the pre-synaptic activity for the trace
            # should be the input to fc1 (x_t), and post-synaptic activity is output of lif1.
            # Our current CustomLIFCell's forward expects input `x` to be pre-synaptic spikes for its *own* trace.
            # This means CustomLIFCell needs to see the spikes that, when multiplied by weights, become its input current.
            # So, the trace in lif1 (hidden_size, hidden_size) is for an implicit recurrent connection if input is output of fc1.
            # OR, if lif1's trace is (input_size_to_fc1, hidden_size_of_lif1), then CustomLIFCell's input_size
            # must match fc1's input_size, and its `forward` needs to be called with `x_t`.
            #
            # Let's adjust CustomLIFCell's design philosophy slightly for this example:
            # The CustomLIFCell is associated with the weights of fc1.
            # Pre-synaptic activity = x_t (input to fc1)
            # Post-synaptic activity = s_lif1 (output of lif1)
            # The eligibility trace should be (fc1.in_features, fc1.out_features)
            # This means CustomLIFCell must be aware of x_t.
            #
            # For now, let's assume lif1.eligibility_trace is (fc1.out_features, fc1.out_features)
            # and it's updated based on lif1_current and s_lif1.
            # To make it (fc1.in_features, fc1.out_features) as desired for fc1's weights,
            # CustomLIFCell's `forward` needs access to `x_t` that went into `fc1`.
            # This requires a slight refactor of CustomLIFCell or how it's called.
            #
            # For this example, let's assume CustomLIFCell is modified or used such that:
            # its `update_trace(pre_syn_activity, post_syn_activity)` is called manually, OR
            # its `forward` is modified to accept `pre_syn_for_trace` explicitly.
            #
            # Simpler approach for now: The `CustomLIFCell` as written takes `x` (its direct input current or spikes)
            # and `s_out` (its output spikes). So its trace is `(input_to_cell, output_of_cell)`.
            # If `lif1` is our custom cell, its trace is `(hidden_size, hidden_size)`.
            # This means the trace is for weights *within* `lif1` (e.g. recurrent) or for `fc1` IF `fc1`'s output is `hidden_size`.
            #
            # Let's make `lif1` have trace `(input_size, hidden_size)` corresponding to `fc1`.
            # This implies `CustomLIFCell` needs to be initialized with `(input_size, hidden_size)`
            # and its `forward` needs to be called with `x_t` (from encoder) as `pre_synaptic_input_for_trace`
            # and its own output spikes as `post_synaptic_activity`.
            # This is a bit tricky with the current `CustomLIFCell` which assumes `x` in `forward(x, state)` is for the trace.
            #
            # Let's redefine `lif1` in init:
            # self.lif1 = CustomLIFCell(input_size=input_size, hidden_size=hidden_size, dt=0.001)
            # Then in forward:
            # lif1_current = self.fc1(x_t)
            # s_lif1, s1 = self.lif1(lif1_current, s1, pre_synaptic_for_trace=x_t) # Modified CustomLIFCell
            #
            # Sticking to current CustomLIFCell:
            # It calculates trace based on its direct input `x` and its output `s_out`.
            # So, if `lif1` is CustomLIFCell(hidden_size, hidden_size), its trace is for weights of size (hidden_size, hidden_size).
            # To apply 3-factor to `fc1`, we need a trace for `fc1`.
            #
            # Option: create a specific `TraceableLinearLIFLayer`
            # For now, let's assume `self.lif1` is a `CustomLIFCell` whose trace we will apply to `self.fc1`.
            # This means `self.lif1` must be initialized with `(input_size, hidden_size)` for its trace.
            # And its `forward` method's `x` argument will be `x_t` (encoder output),
            # and its `s_out` will be the post-synaptic spikes.
            # The LIF dynamics themselves will operate on `self.fc1(x_t)`. This is non-standard.
            #
            # Correct approach for CustomLIFCell as written:
            # If `CustomLIFCell` has `input_size` and `hidden_size` for its trace,
            # and it's applied *after* `fc1`, then `input_size` of cell is `fc1.out_features`.
            # `hidden_size` of cell is `fc1.out_features` (if it's just a cell).
            # The trace is `(fc1.out_features, fc1.out_features)`. This is for recurrent weights.
            #
            # To make the trace for `fc1` (an `nn.Linear`):
            # Trace dimensions: `(fc1.in_features, fc1.out_features)`.
            # Pre-synaptic activity: `x_t` (input to `fc1`).
            # Post-synaptic activity: `s_lif1` (output of `lif1` which processes `fc1(x_t)`).
            #
            # Let's make a conceptual wrapper for `fc1` and `lif1` for clarity in this example.
            # (This will be part of the "SDT Transformer" integration later if relevant)

            # Current pass:
            lif1_current_in = self.fc1(x_t) # Current into LIF neurons
            
            if self.use_custom_lif:
                # To correctly update the trace for fc1 using the current CustomLIFCell:
                # The CustomLIFCell's trace is (cell_input_size, cell_output_size).
                # We want to associate this trace with fc1's weights (input_size, hidden_size).
                # This means the CustomLIFCell instance should be configured as:
                #   `CustomLIFCell(input_size (from encoder), hidden_size (of fc1 output))`
                # And its forward should be called with `x_t` (pre-synaptic to fc1)
                # and it should internally use `lif1_current_in` to compute its own state and spikes.
                # This requires CustomLIFCell.forward to take two types of inputs:
                # one for trace (`x_t`) and one for neuron dynamics (`lif1_current_in`).
                #
                # Let's modify CustomLIFCell slightly for this script or assume it's done.
                # For now, let's assume self.lif1 is the CustomLIFCell and its trace is updated
                # using pre_spikes = x_t and post_spikes = s_lif1.
                # And its internal dynamics use lif1_current_in.
                # This is a common pattern for Hebbian learning at a layer.
                #
                # Simplification for this script: Assume self.lif1 IS CustomLIFCell
                # and it's correctly set up. We'll use its `forward` method.
                # The `x` to `lif1.forward` is `lif1_current_in`.
                # The trace update needs `x_t` (pre to fc1) and `s_lif1` (post from lif1).
                #
                # The current CustomLIFCell's `forward(self, x, state)` uses `x` for BOTH
                # current calculation (implicitly, as it's passed to super().forward) AND trace.
                # This is fine if `fc1` is *inside* the CustomLIFCell.
                # Or, if CustomLIFCell *is* the layer, `x` is input spikes, and it has its own weights.
                #
                # Let's assume CustomLIFCell is just the cell, and fc1 is separate.
                # We need to manually call a trace update method.
                # Add to CustomLIFCell: `update_eligibility_trace(self, pre_syn_spikes, post_syn_spikes)`
                #
                # For this example, I will call `lif1.eligibility_trace.data += ...` directly here for brevity,
                # imagining `lif1` has the trace we need for `fc1`.
                # This means `lif1.eligibility_trace` should be `(input_size, hidden_size)`.
                # And `lif1` itself should be `CustomLIFCell(input_size, hidden_size, ...)`

                s_lif1, s1 = self.lif1(lif1_current_in, s1) # lif1 is CustomLIFCell(hidden_size, hidden_size) by default init
                                                            # so trace is (hidden_size, hidden_size)
                                                            # This trace is for recurrent weights on lif1, not fc1.

                # *** To apply 3-factor to fc1, we need a CustomLIFCell instance specifically for fc1's trace ***
                # Let's assume `self.fc1_trace_cell = CustomLIFCell(input_size, hidden_size)` exists for this.
                # And we call: `self.fc1_trace_cell.update_trace_from_spikes(x_t, s_lif1)`
                # This is getting too complex for a generic training script without refactoring CustomLIFCell.

                # Let's assume `self.lif1` IS the `CustomLIFCell(input_size, hidden_size)`
                # and its `forward` is modified to:
                # `def forward(self, current_input, pre_synaptic_spikes_for_trace, state)`
                # where `current_input` drives neuron dynamics, `pre_synaptic_spikes_for_trace` updates trace.
                # This is the cleanest. I'll simulate this by directly manipulating a trace here.

                # For this script, let's assume self.lif1 *is* the CustomLIFCell,
                # and its trace is (hidden_size, hidden_size) as per its init in this script.
                # We will apply the 3-factor rule to an *implicit* recurrent connection on lif1.
                # This is not what was originally asked (apply to output head or one projection).
                #
                # Let's re-initialize self.lif1 to be for fc1:
                # self.lif1 = CustomLIFCell(input_size, hidden_size)
                # Then in forward:
                #   lif1_current_in = self.fc1(x_t)
                #   # The `x` for CustomLIFCell.forward should be `x_t` for trace,
                #   # but LIF dynamics should use `lif1_current_in`.
                #   # This is the core issue.
                #
                # Path of least modification to CustomLIFCell for this script:
                # We apply the 3-factor rule to fc1.
                # The trace for fc1 is (input_size, hidden_size).
                # Pre-synaptic spikes: x_t
                # Post-synaptic spikes: s_lif1
                # We need a place to store this trace. Let's add it to the model.
                if not hasattr(self, 'fc1_eligibility_trace'):
                    self.fc1_eligibility_trace = torch.zeros_like(self.fc1.weight.data.T) # (in, out) -> (out, in) for .T
                                                                                            # fc1.weight is (out, in)
                                                                                            # trace should be (in, out)
                    self.fc1_eligibility_trace = torch.zeros(self.fc1.in_features, self.fc1.out_features, device=x_static.device)


                # Standard LIF processing for s_lif1
                # (Using a dummy LIF cell here for s_lif1 generation if lif1 is not CustomLIFCell,
                # or if lif1 is CustomLIFCell but we are handling trace manually)
                # For simplicity, assume s_lif1 comes from a standard LIF processing of lif1_current_in
                # This part is tricky without a clear layer definition.
                # Let's use the existing self.lif1 (which is CustomLIFCell if use_custom_lif is True)
                # If it's CustomLIFCell, its own trace is updated. Let's use THAT trace.
                # This means we apply 3-factor to weights of size (lif1.input_size, lif1.hidden_size)
                # which are implicitly part of lif1 if it were a layer, or fc1 if lif1.input_size = fc1.input_size.

                # If self.use_custom_lif: self.lif1 is CustomLIFCell(hidden_size, hidden_size)
                # Its trace is (hidden_size, hidden_size). We'd apply update to a layer with such weights.
                # This is not fc1. This would be for a recurrent connection on the hidden layer.

                # Let's apply to fc1.
                # Pre-activity: x_t (batch, input_size)
                # Post-activity: s_lif1 (batch, hidden_size) - This is output of self.lif1 processing self.fc1(x_t)
                
                # Standard LIF processing using self.lif1 (could be CustomLIFCell or StandardLIFCell)
                # If lif1 is CustomLIFCell, its internal trace is updated based on lif1_current_in and s_lif1.
                # This is fine, we can use that trace if lif1 *is* the layer we target.
                # But we want to target fc1.
                s_lif1, s1 = self.lif1(lif1_current_in, s1)


                # Update fc1_eligibility_trace
                # x_t: (batch, in_feat_fc1)
                # s_lif1: (batch, out_feat_fc1)
                # trace: (in_feat_fc1, out_feat_fc1)
                # Correct CustomLIFCell trace update logic:
                #   batch_trace_update = torch.matmul(pre_spikes.t(), post_spikes) / batch_size
                #   self.eligibility_trace.mul_(decay).add_(batch_trace_update)
                
                # We need to manually manage the trace for fc1 here
                # Assume trace_decay_fc1 = 0.95 (should be a class member)
                if not hasattr(self, 'trace_decay_fc1'): self.trace_decay_fc1 = 0.95

                pre_for_fc1_trace = x_t.detach()
                post_for_fc1_trace = s_lif1.detach() # s_lif1 comes from self.lif1(self.fc1(x_t), ...)
                
                batch_fc1_trace_update = torch.matmul(pre_for_fc1_trace.t(), post_for_fc1_trace) / x_static.shape[0]
                self.fc1_eligibility_trace.mul_(self.trace_decay_fc1).add_(batch_fc1_trace_update)

            else: # Not using custom lif, just standard processing
                s_lif1, s1 = self.lif1(lif1_current_in, s1)


            lif1_output_spikes_over_time.append(s_lif1)
        
        # Stack spikes from lif1
        lif1_output_spikes_over_time = torch.stack(lif1_output_spikes_over_time) # (time, batch, hidden_size)
        
        # For fc2, let's sum spikes over time from lif1 and pass through fc2
        # This is a common way to get a final classification score from spikes
        summed_s_lif1 = torch.sum(lif1_output_spikes_over_time, dim=0) # (batch, hidden_size)
        output = self.fc2(summed_s_lif1) # (batch, output_size)
        
        return output # Return the direct output of fc2

    def reset_fc1_trace(self):
        if hasattr(self, 'fc1_eligibility_trace'):
            self.fc1_eligibility_trace.zero_()


# Hyperparameters
input_size = 50
hidden_size = 100 # fc1 output features, lif1 input/output features
output_size = 10 # e.g., 10 classes
learning_rate_bp = 0.001 # Backprop learning rate
learning_rate_local = 0.005 # Local rule learning rate
num_epochs = 20
batch_size = 32
use_three_factor = True # Ablation switch
clip_local_update = 0.01 # Clipping for local update

# Instantiate model, loss, and optimizer
model = SimpleSNN(input_size, hidden_size, output_size, use_custom_lif=use_three_factor)
criterion = nn.CrossEntropyLoss() # Example loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate_bp)

print(f"Model: input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")
print(f"Using three-factor rule: {use_three_factor}")
if use_three_factor:
    print(f"  Local LR: {learning_rate_local}, Clip: {clip_local_update}")
    # Check if fc1_eligibility_trace will be created
    if not model.use_custom_lif :
        print("Warning: use_custom_lif is False, so fc1_eligibility_trace might not be used as expected by three_factor_updater.")
        print("The current setup for three-factor plasticity manually creates and updates fc1_eligibility_trace "
              "if use_custom_lif is True, based on x_t and s_lif1.")
    # The flag `use_custom_lif` in SimpleSNN is now a bit misleading.
    # It controls whether `lif1` is CustomLIFCell or StandardLIFCell.
    # The manual trace update for fc1 is gated by `if self.use_custom_lif:` block in forward.
    # Let's rename it to `enable_plasticity_related_traces` for clarity.
    # For now, we'll proceed with `use_custom_lif` as the switch for enabling the fc1 trace accumulation.


# Training loop
for epoch in range(num_epochs):
    # Dummy data for each epoch
    # Static features, encoder will create spike trains
    dummy_static_input = torch.rand(batch_size, input_size) 
    dummy_labels = torch.randint(0, output_size, (batch_size,))

    # Reset traces if applicable (e.g., at the start of a trajectory/batch)
    if use_three_factor and hasattr(model, 'reset_fc1_trace'):
        model.reset_fc1_trace()
    # If model.lif1 is a CustomLIFCell and its trace is used, reset it too:
    if model.use_custom_lif and isinstance(model.lif1, CustomLIFCell):
         model.lif1.reset_trace()


    # Forward pass
    outputs = model(dummy_static_input)

    # Calculate loss (standard backprop part)
    loss = criterion(outputs, dummy_labels)

    # Backward pass and optimize (standard backprop)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Three-factor local update (if enabled)
    if use_three_factor:
        # 1. Get the eligibility trace for fc1
        #    This trace was accumulated during the forward pass.
        if not hasattr(model, 'fc1_eligibility_trace'):
            print("Warning: fc1_eligibility_trace not found on model. Skipping three-factor update.")
        else:
            trace_for_fc1 = model.fc1_eligibility_trace

            # 2. Determine Return-to-Go (G_t)
            #    This is a placeholder. In a real RL task, G_t would be calculated based on rewards.
            #    For a supervised task, it could be related to the error or a success signal.
            #    Let's use a dummy scalar G_t. If loss is low, G_t is positive (rewarding).
            #    This is a simplification; G_t is usually time-dependent. Here, one G_t for the batch.
            g_t = 1.0 / (loss.item() + 1e-2) # Example: higher reward for lower loss
            g_t_tensor = torch.tensor(g_t, device=model.fc1.weight.device)
            
            # 3. Apply the update to fc1
            apply_three_factor_update(
                layer_to_update=model.fc1, # The nn.Linear layer
                eligibility_trace=trace_for_fc1, # Shape: (input_size, hidden_size)
                return_to_go=g_t_tensor,
                local_learning_rate=learning_rate_local,
                clip_value=clip_local_update
            )
            # print(f"Applied 3-factor update to fc1. G_t={g_t:.2f}")


    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training finished.")

# Ablation & Validation suggestion:
# To measure training speed, you would typically run this loop multiple times,
# once with use_three_factor = True and once with use_three_factor = False.
# Then compare how many epochs it takes to reach a certain loss value or accuracy
# on a validation set.

# Example: How to access the trace from CustomLIFCell if it were used directly
if model.use_custom_lif and isinstance(model.lif1, CustomLIFCell):
    # model.lif1 is CustomLIFCell(hidden_size, hidden_size) in this setup
    # Its trace is (hidden_size, hidden_size), for its own implicit weights.
    # This trace is NOT for fc1 unless lif1 was CustomLIFCell(input_size, hidden_size)
    # and its forward method was structured to use separate inputs for dynamics and trace.
    # trace_from_custom_lif1 = model.lif1.eligibility_trace
    # print(f"\nEligibility trace from model.lif1 (CustomLIFCell of size ({model.lif1.input_size},{model.lif1.hidden_size})) after training:\n{trace_from_custom_lif1.shape}")
    pass

print(f"\nFinal eligibility trace for fc1 (shape {model.fc1_eligibility_trace.shape if hasattr(model, 'fc1_eligibility_trace') else 'N/A'}):")
if hasattr(model, 'fc1_eligibility_trace'):
    # print(model.fc1_eligibility_trace)
    pass

print("\nScript execution complete.")
print("Note: The integration of eligibility trace calculation for `fc1` was done manually in the model's forward pass.")
print("A cleaner way would be to have a dedicated layer type e.g., `TraceableLinearLIF` or modify `CustomLIFCell`")
print("to explicitly take `pre_synaptic_spikes_for_trace` in its `forward` method, separate from the input that drives its current.")

# Small fix for apply_three_factor_update import if it's not found due to path issues
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from three_factor_updater import apply_three_factor_update # Already imported

# Final check on CustomLIFCell's trace logic:
# CustomLIFCell's trace is `(self.input_size, self.hidden_size)`.
# In `SimpleSNN`, `self.lif1 = CustomLIFCell(hidden_size, hidden_size)`.
# So, `self.lif1.eligibility_trace` is `(hidden_size, hidden_size)`.
# This trace is suitable for a recurrent connection on the `lif1` layer neurons,
# or for a linear layer of size `(hidden_size, hidden_size)` that inputs to `lif1`.
#
# The manual trace `self.fc1_eligibility_trace` is `(input_size, hidden_size)`,
# correctly corresponding to `self.fc1.weight`. This is the one we are using for `apply_three_factor_update`.
# The current script correctly uses this manually managed trace for `fc1`.
# The `use_custom_lif` flag is used to decide if `self.lif1` is `CustomLIFCell` or `StandardLIFCell`,
# and also gates the manual accumulation of `self.fc1_eligibility_trace`. This is a bit coupled but works for this example.
# A more decoupled approach would separate the choice of cell type from the decision to use plasticity on fc1.
# For example, an additional flag like `model.enable_fc1_plasticity = True`.
# The current setup is functional for demonstrating the concept.
