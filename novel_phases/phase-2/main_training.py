import torch
import torch.nn as nn
import torch.optim as optim
from norse.torch import LIFState, PoissonEncoder 
from custom_lif import CustomLIFCell 
from three_factor_updater import apply_three_factor_update 
import numpy as np # For storing and analyzing losses

# For reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Define a simple SNN model (same as before)
class SimpleSNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_three_factor_rule_active=True): # Renamed for clarity
        super(SimpleSNN, self).__init__()
        self.use_three_factor_rule_active = use_three_factor_rule_active # This flag enables trace accumulation
        self.encoder = PoissonEncoder(seq_length=10) 

        self.fc1 = nn.Linear(input_size, hidden_size)
        if self.use_three_factor_rule_active: # This controls if LIF1 is Custom or Standard, not directly trace for FC1
            # The actual trace for fc1 is manually managed if use_three_factor_rule_active is true later
            self.lif1 = CustomLIFCell(input_size=hidden_size, hidden_size=hidden_size, dt=0.001)
        else:
            from norse.torch import LIFCell as StandardLIFCell 
            self.lif1 = StandardLIFCell(p=None, dt=0.001)

        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Trace related attributes for fc1, managed if use_three_factor_rule_active
        self.fc1_eligibility_trace = None
        self.trace_decay_fc1 = 0.95


    def forward(self, x_static): 
        spikes_over_time = self.encoder(x_static) 
        s1 = None 
        lif1_output_spikes_over_time = []

        # Initialize fc1_eligibility_trace here if it's the first pass for this model instance
        # and if plasticity is conceptually active for fc1
        if self.use_three_factor_rule_active and self.fc1_eligibility_trace is None:
             self.fc1_eligibility_trace = torch.zeros(
                self.fc1.out_features, self.fc1.in_features, 
                device=self.fc1.weight.device, dtype=self.fc1.weight.dtype
            )

        for t in range(spikes_over_time.shape[0]): 
            x_t = spikes_over_time[t] 
            lif1_current_in = self.fc1(x_t) 
            s_lif1, s1 = self.lif1(lif1_current_in, s1)

            if self.use_three_factor_rule_active and self.fc1_eligibility_trace is not None:
                pre_for_fc1_trace = x_t.detach() 
                post_for_fc1_trace = s_lif1.detach() 
                batch_fc1_trace_update = torch.matmul(post_for_fc1_trace.t(), pre_for_fc1_trace) / x_static.shape[0]
                self.fc1_eligibility_trace.mul_(self.trace_decay_fc1).add_(batch_fc1_trace_update)
            
            lif1_output_spikes_over_time.append(s_lif1)
        
        lif1_output_spikes_over_time = torch.stack(lif1_output_spikes_over_time)
        summed_s_lif1 = torch.sum(lif1_output_spikes_over_time, dim=0) 
        output = self.fc2(summed_s_lif1) 
        return output

    def reset_fc1_trace(self):
        if self.fc1_eligibility_trace is not None:
            self.fc1_eligibility_trace.zero_()

# --- Training Function ---
def train_model(model, criterion, optimizer, num_epochs, batch_size, input_size, output_size, 
                device, current_local_learning_rate, clip_local_update, 
                print_metrics=False):
    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        # Reset traces at the start of each epoch
        if hasattr(model, 'reset_fc1_trace') and model.use_three_factor_rule_active and current_local_learning_rate > 0:
            model.reset_fc1_trace()
        if isinstance(model.lif1, CustomLIFCell) and model.use_three_factor_rule_active and current_local_learning_rate > 0:
             model.lif1.reset_trace()

        # Dummy data generation for each epoch for simplicity in this example
        # In a real scenario, you'd use a DataLoader
        for _ in range(50): # Simulate 50 batches per epoch
            dummy_static_input = torch.rand(batch_size, input_size, device=device) 
            dummy_labels = torch.randint(0, output_size, (batch_size,), device=device)
            num_batches += 1

            outputs = model(dummy_static_input)
            loss = criterion(outputs, dummy_labels)

            optimizer.zero_grad()
            loss.backward()
            
            # --- Metrics: Gradient Norm (for fc1) ---
            fc1_grad_norm = None
            if model.fc1.weight.grad is not None:
                fc1_grad_norm = model.fc1.weight.grad.norm().item()
            
            optimizer.step()
            running_loss += loss.item()

            # --- Three-factor local update (if enabled by current_local_learning_rate) ---
            local_update_norm = None
            if model.use_three_factor_rule_active and current_local_learning_rate > 0:
                if model.fc1_eligibility_trace is not None:
                    trace_for_fc1 = model.fc1_eligibility_trace
                    g_t = 1.0 / (loss.item() + 1e-2) 
                    g_t_tensor = torch.tensor(g_t, device=model.fc1.weight.device)
                    
                    delta_W = apply_three_factor_update(
                        layer_to_update=model.fc1, 
                        eligibility_trace=trace_for_fc1, 
                        return_to_go=g_t_tensor,
                        local_learning_rate=current_local_learning_rate,
                        clip_value=clip_local_update
                    )
                    if delta_W is not None:
                        local_update_norm = delta_W.norm().item()
                else:
                    if print_metrics and (epoch + 1) % (num_epochs // 4) == 0 : # Print less frequently
                         print(f"Epoch {epoch+1}: fc1_eligibility_trace is None, skipping 3-factor update.")


            if print_metrics and (epoch + 1) % (num_epochs // 4) == 0 and _ == 0 : # Print for first batch of milestone epochs
                print(f"Epoch [{epoch+1}/{num_epochs}] (Batch 1): Loss: {loss.item():.4f}")
                if fc1_grad_norm is not None:
                    print(f"  FC1 Grad Norm: {fc1_grad_norm:.4e}")
                if local_update_norm is not None:
                    print(f"  Local Update Norm (FC1): {local_update_norm:.4e}")
                if model.use_three_factor_rule_active and current_local_learning_rate > 0 and model.fc1_eligibility_trace is not None:
                    trace_stats = (model.fc1_eligibility_trace.mean().item(), model.fc1_eligibility_trace.std().item(),
                                   model.fc1_eligibility_trace.min().item(), model.fc1_eligibility_trace.max().item())
                    print(f"  FC1 Eligibility Trace Stats (Mean/Std/Min/Max): {trace_stats[0]:.2e} / {trace_stats[1]:.2e} / {trace_stats[2]:.2e} / {trace_stats[3]:.2e}")
        
        epoch_avg_loss = running_loss / num_batches
        epoch_losses.append(epoch_avg_loss)
        if (epoch + 1) % (num_epochs // 10) == 0 : # Print epoch average loss less frequently
             print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_avg_loss:.4f}")
             
    return epoch_losses


# --- Hyperparameters & Setup ---
input_size = 50
hidden_size = 100 
output_size = 10 
learning_rate_bp = 0.001 
base_local_learning_rate = 0.005 # For the run WITH plasticity
num_epochs = 100 
batch_size = 32
clip_local_update = 0.01 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print(f"Model: input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")
print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, BP LR: {learning_rate_bp}\n")

# --- Run 1: With Three-Factor Update ---
print("--- Training: WITH Three-Factor Updates ---")
print(f"Local LR: {base_local_learning_rate}, Clip: {clip_local_update}")
# Model instantiation: use_three_factor_rule_active=True enables trace mechanism
model_with_3f = SimpleSNN(input_size, hidden_size, output_size, use_three_factor_rule_active=True).to(device)
criterion_with_3f = nn.CrossEntropyLoss()
optimizer_with_3f = optim.Adam(model_with_3f.parameters(), lr=learning_rate_bp)

losses_with_3f = train_model(
    model=model_with_3f, criterion=criterion_with_3f, optimizer=optimizer_with_3f,
    num_epochs=num_epochs, batch_size=batch_size, input_size=input_size, output_size=output_size,
    device=device, current_local_learning_rate=base_local_learning_rate, 
    clip_local_update=clip_local_update, print_metrics=True
)
print("--- Finished Training WITH Three-Factor Updates ---\n")


# --- Run 2: WITHOUT Three-Factor Update (local_lr = 0) ---
print("--- Training: WITHOUT Three-Factor Updates (Ablation) ---")
# Model instantiation: use_three_factor_rule_active=False disables trace accumulation and CustomLIF
# For a fair comparison of only the update rule, we should use the same model architecture (with trace mechanism enabled)
# but set local_learning_rate to 0.
# If use_three_factor_rule_active is False, fc1_eligibility_trace is never created.
model_no_3f = SimpleSNN(input_size, hidden_size, output_size, use_three_factor_rule_active=True).to(device) # Still True to allow trace mechanism for fair comparison
criterion_no_3f = nn.CrossEntropyLoss()
optimizer_no_3f = optim.Adam(model_no_3f.parameters(), lr=learning_rate_bp)

losses_no_3f = train_model(
    model=model_no_3f, criterion=criterion_no_3f, optimizer=optimizer_no_3f,
    num_epochs=num_epochs, batch_size=batch_size, input_size=input_size, output_size=output_size,
    device=device, current_local_learning_rate=0.0, # Key change for ablation
    clip_local_update=clip_local_update, print_metrics=False # Less verbose for ablation run
)
print("--- Finished Training WITHOUT Three-Factor Updates ---\n")


# --- Output for Plotting ---
print("Loss data for plotting (copy these arrays):")
print(f"losses_with_3f = {losses_with_3f}")
print(f"losses_no_3f = {losses_no_3f}")
print("\nTo plot, use a library like matplotlib in your local environment or Kaggle notebook:")
print("Example Matplotlib code:")
print("import matplotlib.pyplot as plt")
print("epochs_range = range(1, num_epochs + 1)")
print("plt.figure(figsize=(10, 6))")
print("plt.plot(epochs_range, losses_with_3f, label='With 3-Factor Update (Local LR > 0)')")
print("plt.plot(epochs_range, losses_no_3f, label='Without 3-Factor Update (Local LR = 0)')")
print("plt.xlabel('Epochs')")
print("plt.ylabel('Average Loss')")
print("plt.title('Training Loss Comparison')")
print("plt.legend()")
print("plt.grid(True)")
print("plt.show()")

print("\nScript execution complete.")
print("Review printed metrics for gradient norms, local update norms, and trace statistics during the 'WITH Three-Factor Updates' run.")
