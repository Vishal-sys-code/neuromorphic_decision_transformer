import torch
import torch.nn as nn
import sys
import os
import copy

# --- Path Setup --- 
# Assuming the script is in project_root/novel_phases/phase-2/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

src_path = os.path.join(project_root, 'src')
external_submodule_path = os.path.join(project_root, 'external') 

if src_path not in sys.path:
    sys.path.insert(0, src_path)
if external_submodule_path not in sys.path:
    sys.path.insert(0, external_submodule_path)

try:
    from src.models.snn_dt import SNNDecisionTransformer 
except ImportError as e:
    print(f"Import Error: {e}.\nEnsure 'src' and 'external' directories are correctly structured and accessible.")
    print(f"Current sys.path includes: {project_root}, {src_path}, {external_submodule_path}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    sys.exit(1)

# --- Test Configuration --- 
STATE_DIM, ACT_DIM, HIDDEN_SIZE = 4, 2, 128 
MAX_LENGTH, MAX_EP_LEN = 10, 100 
N_LAYER, N_HEAD = 3, 1 
BATCH_SIZE, SEQ_LENGTH = 2, 5 

# --- Helper Functions --- 
def create_dummy_input(device, batch_size=BATCH_SIZE, seq_length=SEQ_LENGTH, make_rtg_positive=False):
    states = torch.rand(batch_size, seq_length, STATE_DIM, device=device)
    actions = torch.rand(batch_size, seq_length, ACT_DIM, device=device) 
    rewards = torch.rand(batch_size, seq_length, 1, device=device) 
    if make_rtg_positive:
        returns_to_go = torch.rand(batch_size, seq_length, 1, device=device) + 0.5 # Ensure positive and non-trivial
    else:
        returns_to_go = torch.rand(batch_size, seq_length, 1, device=device)
    timesteps = torch.randint(0, MAX_EP_LEN, (batch_size, seq_length), device=device)
    attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)
    return states, actions, rewards, returns_to_go, timesteps, attention_mask

# --- Test Execution --- 
def run_new_plasticity_tests():
    print("--- Starting SNNDT Three-Factor Rule Test ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_core_args = {
        'state_dim': STATE_DIM, 'act_dim': ACT_DIM, 'hidden_size': HIDDEN_SIZE,
        'max_length': MAX_LENGTH, 'max_ep_len': MAX_EP_LEN,
        'n_layer': N_LAYER, 'n_head': N_HEAD, 'action_tanh': False,
        'n_inner': 4 * HIDDEN_SIZE, 'activation_function': 'gelu',
        'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
    }
    snn_action_head_args = {
        'time_window': 10,       
        'lif_threshold': 0.05,   
        'action_lr': 1e-3        
    }
    model_args = {**model_core_args, **snn_action_head_args}

    try:
        model = SNNDecisionTransformer(**model_args).to(device)
        print("SNNDT with new three-factor rule action head instantiated successfully.")
    except Exception as e:
        print(f"ERROR: Instantiation Failed: {type(e).__name__}: {e}")
        return 

    print("\n--- Test 1: Weight Update Check ---")
    model.train() 
    dummy_inputs = create_dummy_input(device, make_rtg_positive=True)
    states, actions, rewards, returns_to_go, timesteps, attention_mask = dummy_inputs
    initial_weights = model.action_lin.weight.data.clone()

    print(f"Performing forward pass 1 (Batch Counter: {model.batch_counter})...")
    _ = model(states, actions, rewards, returns_to_go, timesteps, attention_mask)
    print("Forward pass 1 complete.")

    weights_after_pass1 = model.action_lin.weight.data.clone()
    weights_changed = not torch.equal(initial_weights, weights_after_pass1)
    
    print(f"Initial action_lin weight norm: {torch.norm(initial_weights).item():.4f}")
    print(f"Action_lin weight norm after pass 1: {torch.norm(weights_after_pass1).item():.4f}")
    assert weights_changed, (
        "FAIL: action_lin weights did not change after the first forward pass in training mode. "
        "Check model's diagnostic prints for spike activity and delta_W values. Ensure RTG are non-zero."
    )
    print("PASS: action_lin weights changed after first pass.")

    print("\n--- Test 2: Batch Counter & Subsequent Passes Check ---")
    for i in range(4):
        current_batch_count = model.batch_counter 
        print(f"Performing forward pass {i+2} (Batch Counter before this pass: {current_batch_count})...")
        _ = model(states, actions, rewards, returns_to_go, timesteps, attention_mask)
        print(f"Forward pass {i+2} complete. Batch counter is now {model.batch_counter}.")
        if current_batch_count < 5:
            assert model.batch_counter == current_batch_count + 1, f"FAIL: Batch counter did not increment. Was {current_batch_count}, now {model.batch_counter}"
        else:
            assert model.batch_counter == 5, f"FAIL: Batch counter > 5. Was {current_batch_count}, now {model.batch_counter}"
    print("PASS: Batch counter behavior seems okay.")

    print("\n--- Test 3: Eval Mode Check ---")
    model.eval()
    weights_before_eval_pass = model.action_lin.weight.data.clone()
    print(f"Performing forward pass in eval mode (Batch Counter: {model.batch_counter})...")
    try:
        _ = model(states, actions, rewards, returns_to_go, timesteps, attention_mask)
        print("Forward pass in eval mode successful.")
    except Exception as e:
        assert False, f"FAIL: Model errored during forward pass in eval mode: {type(e).__name__}: {e}"
    
    weights_after_eval_pass = model.action_lin.weight.data.clone()
    assert torch.equal(weights_before_eval_pass, weights_after_eval_pass), (
        "FAIL: action_lin weights changed during forward pass in eval mode."
    )
    print("PASS: action_lin weights did not change in eval mode.")

    del model
    print("\nModel deleted.")
    print("--- SNNDT Three-Factor Rule Test Completed Successfully! ---")
    print("Review console output for detailed diagnostic messages from the model regarding spikes and delta_W.")

if __name__ == "__main__":
    run_new_plasticity_tests()