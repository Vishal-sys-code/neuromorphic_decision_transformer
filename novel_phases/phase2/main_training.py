"""
Authored by: Vishal Pandey
Reviewed by: Debasmita Biswas
"""
import torch
import torch.nn as nn
import torch.optim as optim
from norse.torch import LIFState, PoissonEncoder 
from custom_lif import CustomLIFCell 
from three_factor_updater import apply_three_factor_update 
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

class SimpleSNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_three_factor_rule_active=True): # print_signal_stats removed, will be passed to forward
        super(SimpleSNN, self).__init__()
        self.use_three_factor_rule_active = use_three_factor_rule_active
        self.encoder = PoissonEncoder(seq_length=10) 
        self.fc1 = nn.Linear(input_size, hidden_size)
        if self.use_three_factor_rule_active: 
            self.lif1 = CustomLIFCell(input_size=hidden_size, hidden_size=hidden_size, dt=0.001)
        else:
            from norse.torch import LIFCell as StandardLIFCell 
            self.lif1 = StandardLIFCell(p=None, dt=0.001)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc1_eligibility_trace = None
        self.trace_decay_fc1 = 0.95
        
    # Removed _signal_stats_printed_this_epoch and new_epoch_reset, 
    # print_signal_stats_for_this_batch will be passed to forward by train_model

    def forward(self, x_static, print_signal_stats_for_this_batch=False, current_epoch_for_signal_print=-1, current_batch_idx_for_signal_print=-1): 
        spikes_over_time = self.encoder(x_static) 
        s1 = None 
        lif1_output_spikes_over_time = []

        if self.use_three_factor_rule_active and self.fc1_eligibility_trace is None:
             self.fc1_eligibility_trace = torch.zeros(
                self.fc1.out_features, self.fc1.in_features, 
                device=self.fc1.weight.device, dtype=self.fc1.weight.dtype
            )

        for t in range(spikes_over_time.shape[0]): 
            x_t = spikes_over_time[t] 
            lif1_current_in = self.fc1(x_t) 
            s_lif1, s1 = self.lif1(lif1_current_in, s1) 

            if print_signal_stats_for_this_batch and t < 3: # Print for first 3 timesteps if flag is true
                pre_act = x_t.detach().cpu()
                post_act = s_lif1.detach().cpu()
                # Ensure epoch/batch numbers are valid if passed for printing context
                epoch_str = f"Epoch {current_epoch_for_signal_print}, " if current_epoch_for_signal_print != -1 else ""
                batch_str = f"Batch {current_batch_idx_for_signal_print}, " if current_batch_idx_for_signal_print != -1 else ""
                print(f"  {epoch_str}{batch_str}t={t}:")
                print(f"    x_t (pre-fc1) stats: mean={pre_act.mean():.2e}, std={pre_act.std():.2e}, min={pre_act.min():.2e}, max={pre_act.max():.2e}, sum={pre_act.sum().item():.1f}")
                print(f"    s_lif1 (post-lif1) stats: mean={post_act.mean():.2e}, std={post_act.std():.2e}, min={post_act.min():.2e}, max={post_act.max():.2e}, sum={post_act.sum().item():.1f}")

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

def train_model(model, criterion, optimizer, num_epochs, batch_size, input_size, output_size, 
                device, current_local_learning_rate, clip_local_update, 
                enable_detailed_metrics_printing=False, run_label=""): # Renamed print_metrics
    epoch_losses = []
    print(f"\n--- Starting Training Run: {run_label} ---")
    print(f"Local LR: {current_local_learning_rate}, Clip: {clip_local_update if current_local_learning_rate > 0 else 'N/A'}")

    # Determine epochs for detailed metric printing (approx 10 times)
    print_interval = max(1, num_epochs // 10) # Ensure interval is at least 1

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        # Flag to print detailed metrics for the first batch of this specific epoch
        print_detailed_metrics_this_epoch_first_batch = enable_detailed_metrics_printing and (epoch + 1) % print_interval == 0

        if hasattr(model, 'reset_fc1_trace') and model.use_three_factor_rule_active and current_local_learning_rate > 0:
            model.reset_fc1_trace()

        for batch_idx in range(50): 
            dummy_static_input = torch.rand(batch_size, input_size, device=device) 
            dummy_labels = torch.randint(0, output_size, (batch_size,), device=device)
            num_batches += 1

            # Control printing of signal stats: only for the first batch of a metrics epoch
            should_print_signal_stats_for_this_batch = print_detailed_metrics_this_epoch_first_batch and batch_idx == 0
            
            outputs = model(dummy_static_input, 
                            print_signal_stats_for_this_batch=should_print_signal_stats_for_this_batch,
                            current_epoch_for_signal_print=epoch+1, 
                            current_batch_idx_for_signal_print=batch_idx)
            loss = criterion(outputs, dummy_labels)

            optimizer.zero_grad()
            loss.backward()
            
            fc1_grad_sample = None
            fc1_grad_norm = None
            if model.fc1.weight.grad is not None:
                fc1_grad_norm = model.fc1.weight.grad.norm().item()
                fc1_grad_sample = model.fc1.weight.grad.detach().cpu().numpy()[0, :min(5, model.fc1.in_features)]
            
            optimizer.step()
            running_loss += loss.item()

            local_update_norm = None
            local_delta_W_sample = None
            g_t_value = "N/A"
            if model.use_three_factor_rule_active and current_local_learning_rate > 0:
                if model.fc1_eligibility_trace is not None:
                    trace_for_fc1 = model.fc1_eligibility_trace
                    g_t = 1.0 if loss.item() < 1.0 else -0.2 
                    g_t_value = g_t 
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
                        local_delta_W_sample = delta_W.detach().cpu().numpy()[0, :min(5, model.fc1.in_features)]
                elif print_detailed_metrics_this_epoch_first_batch and batch_idx == 0: # Only print warning once per metrics epoch
                     print(f"Epoch {epoch+1} ({run_label}): fc1_eligibility_trace is None, skipping 3-factor update.")

            if print_detailed_metrics_this_epoch_first_batch and batch_idx == 0 : 
                print(f"Epoch [{epoch+1}/{num_epochs}] ({run_label}, Batch 1): Loss: {loss.item():.4f}, G_t: {g_t_value}")
                if fc1_grad_norm is not None:
                    print(f"  FC1 Grad Norm: {fc1_grad_norm:.4e}")
                    if fc1_grad_sample is not None : print(f"  FC1 Grad Sample (1st out, :5): {fc1_grad_sample}")
                if local_update_norm is not None:
                    print(f"  Local Update Norm (FC1): {local_update_norm:.4e}")
                    if local_delta_W_sample is not None : print(f"  Local Delta_W Sample (1st out, :5): {local_delta_W_sample}")
                if model.use_three_factor_rule_active and current_local_learning_rate > 0 and model.fc1_eligibility_trace is not None:
                    trace = model.fc1_eligibility_trace.detach().cpu()
                    trace_stats = (trace.mean().item(), trace.std().item(), trace.min().item(), trace.max().item())
                    print(f"  FC1 Eligibility Trace Stats (Mean/Std/Min/Max): {trace_stats[0]:.2e} / {trace_stats[1]:.2e} / {trace_stats[2]:.2e} / {trace_stats[3]:.2e}")
        
        epoch_avg_loss = running_loss / num_batches
        epoch_losses.append(epoch_avg_loss)
        # Print epoch average loss at the same interval as detailed metrics for consistency
        if (epoch + 1) % print_interval == 0 : 
             print(f"Epoch [{epoch+1}/{num_epochs}] ({run_label}) Average Loss: {epoch_avg_loss:.4f}")
    print(f"--- Finished Training Run: {run_label} ---")
    return epoch_losses

input_size = 50
hidden_size = 100 
output_size = 10 
learning_rate_bp = 0.001 
base_local_learning_rate = 0.25 
clip_local_update = 0.5      

num_epochs = 100 
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print(f"Model: input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")
print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, BP LR: {learning_rate_bp}\n")

# --- Run 1: With Three-Factor Update ---
model_with_3f = SimpleSNN(input_size, hidden_size, output_size, use_three_factor_rule_active=True).to(device)
criterion_with_3f = nn.CrossEntropyLoss()
optimizer_with_3f = optim.Adam(model_with_3f.parameters(), lr=learning_rate_bp)

losses_with_3f = train_model(
    model=model_with_3f, criterion=criterion_with_3f, optimizer=optimizer_with_3f,
    num_epochs=num_epochs, batch_size=batch_size, input_size=input_size, output_size=output_size,
    device=device, current_local_learning_rate=base_local_learning_rate, 
    clip_local_update=clip_local_update, enable_detailed_metrics_printing=True, run_label="With 3-Factor"
)

# --- Run 2: WITHOUT Three-Factor Update (local_lr = 0) ---
model_no_3f = SimpleSNN(input_size, hidden_size, output_size, use_three_factor_rule_active=True).to(device) 
criterion_no_3f = nn.CrossEntropyLoss()
optimizer_no_3f = optim.Adam(model_no_3f.parameters(), lr=learning_rate_bp)

losses_no_3f = train_model(
    model=model_no_3f, criterion=criterion_no_3f, optimizer=optimizer_no_3f,
    num_epochs=num_epochs, batch_size=batch_size, input_size=input_size, output_size=output_size,
    device=device, current_local_learning_rate=0.0, 
    clip_local_update=clip_local_update, enable_detailed_metrics_printing=False, run_label="Without 3-Factor (Ablation)"
)

print("\n--- Ablation Study Results ---")
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
print("Review printed metrics for gradient norms, local update norms, trace statistics, and pre/post signal stats during the 'WITH Three-Factor Updates' run.")
