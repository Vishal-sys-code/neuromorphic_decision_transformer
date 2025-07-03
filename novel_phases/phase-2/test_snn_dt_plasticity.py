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

# Add src and external to sys.path more directly if SpikingMindRL is the project_root
# This handles cases where SpikingMindRL might be a sub-directory or the root itself.
src_path = os.path.join(project_root, 'src')
external_submodule_path = os.path.join(project_root, 'external') # Path to the 'external' directory itself

if src_path not in sys.path:
    sys.path.insert(0, src_path)
if external_submodule_path not in sys.path:
    sys.path.insert(0, external_submodule_path)

# Ensure submodules within 'external' like 'decision-transformer' are also accessible if needed by imports
# For instance, if 'external/decision_transformer/gym/...' is directly imported elsewhere.
# The SNNDecisionTransformer itself handles its relative import for the base class correctly.

try:
    from src.models.snn_dt import SNNDecisionTransformer # Should now work if src_path is correct
    # from transformers import GPT2Config # Not directly used in this test script for model config
except ImportError as e:
    print(f"Import Error: {e}.\nEnsure 'src' and 'external' directories are correctly structured and accessible.")
    print(f"Current sys.path includes: {project_root}, {src_path}, {external_submodule_path}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    sys.exit(1)

# --- Test Configuration --- 
STATE_DIM, ACT_DIM, HIDDEN_SIZE = 4, 2, 128 # Standard DT dimensions
MAX_LENGTH, MAX_EP_LEN = 10, 100 # Max sequence length for transformer, max episode length for embeddings
N_LAYER, N_HEAD = 3, 1 # Transformer config
BATCH_SIZE, SEQ_LENGTH = 2, 5 # Test batch and sequence dimensions

# --- Helper Functions --- 
def create_dummy_input(device, batch_size=BATCH_SIZE, seq_length=SEQ_LENGTH, make_rtg_positive=False):
    states = torch.rand(batch_size, seq_length, STATE_DIM, device=device)
    actions = torch.rand(batch_size, seq_length, ACT_DIM, device=device) # Actual actions taken
    rewards = torch.rand(batch_size, seq_length, 1, device=device) 
    if make_rtg_positive:
        # Ensure returns_to_go are positive and non-trivial for testing weight updates
        returns_to_go = torch.rand(batch_size, seq_length, 1, device=device) + 0.1
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

    # Standard Decision Transformer arguments
    model_core_args = {
        'state_dim': STATE_DIM, 'act_dim': ACT_DIM, 'hidden_size': HIDDEN_SIZE,
        'max_length': MAX_LENGTH, 'max_ep_len': MAX_EP_LEN,
        'n_layer': N_LAYER, 'n_head': N_HEAD, 'action_tanh': False,
        'n_inner': 4 * HIDDEN_SIZE, 'activation_function': 'gelu',
        'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
    }

    # New parameters for our modified SNNDecisionTransformer's action head
    snn_action_head_args = {
        'time_window': 10,       # For SpikingTransformerBlocks (inherited usage)
        'lif_threshold': 0.05,   # Lowered threshold to encourage spiking for testing
        'action_lr': 1e-3        # Learning rate for the action_lin weights
    }
    
    model_args = {**model_core_args, **snn_action_head_args}

    try:
        model = SNNDecisionTransformer(**model_args).to(device)
        print("SNNDT with new three-factor rule action head instantiated successfully.")
    except Exception as e:
        print(f"ERROR: Instantiation Failed: {type(e).__name__}: {e}")
        return # Exit test if model can't be created

    # Test 1: Check weight changes after a forward pass in training mode
    print("\n--- Test 1: Weight Update Check ---")
    model.train() # Ensure model is in training mode for updates
    
    # Create inputs designed to likely cause weight updates (positive RTG)
    dummy_inputs = create_dummy_input(device, make_rtg_positive=True)
    states, actions, rewards, returns_to_go, timesteps, attention_mask = dummy_inputs

    initial_weights = model.action_lin.weight.data.clone()

    print(f"Performing forward pass 1 (Batch Counter: {model.batch_counter})...")
    # The SNNDecisionTransformer's forward method will print spike/delta_W diagnostics
    _ = model(states, actions, rewards, returns_to_go, timesteps, attention_mask)
    print("Forward pass 1 complete.")

    weights_after_pass1 = model.action_lin.weight.data.clone()
    weights_changed = not torch.equal(initial_weights, weights_after_pass1)
    
    print(f"Initial action_lin weight norm: {torch.norm(initial_weights).item():.4f}")
    print(f"Action_lin weight norm after pass 1: {torch.norm(weights_after_pass1).item():.4f}")
    assert weights_changed, (
        "FAIL: action_lin weights did not change after the first forward pass in training mode. "
        "Check for spikes and non-zero returns_to_go. Diagnostics from model should indicate activity."
    )
    print("PASS: action_lin weights changed after first pass.")

    # Test 2: Check batch_counter increment and further diagnostic prints
    print("\n--- Test 2: Batch Counter & Subsequent Passes Check ---")
    # Expect batch_counter to have incremented if it was < 5
    # Perform a few more passes to see diagnostics if batch_counter is still low
    for i in range(4):
        current_batch_count = model.batch_counter # Get counter before potential increment
        print(f"Performing forward pass {i+2} (Batch Counter before this pass: {current_batch_count})...")
        _ = model(states, actions, rewards, returns_to_go, timesteps, attention_mask)
        print(f"Forward pass {i+2} complete. Batch counter is now {model.batch_counter}.")
        # We expect the counter to increment until it reaches 5, then stop incrementing for these checks.
        if current_batch_count < 5:
            assert model.batch_counter == current_batch_count + 1, f"FAIL: Batch counter did not increment as expected. Was {current_batch_count}, now {model.batch_counter}"
        else:
            assert model.batch_counter == 5, f"FAIL: Batch counter expected to stay at 5, but is {model.batch_counter}"
    print("PASS: Batch counter behavior and multiple forward passes seem okay (check console for diagnostics).")

    # Test 3: Ensure model runs in eval mode without error (and no weight updates)
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
        "FAIL: action_lin weights changed during forward pass in eval mode. Updates should only occur in training mode."
    )
    print("PASS: action_lin weights did not change in eval mode.")

    del model
    print("\nModel deleted.")
    print("--- SNNDT Three-Factor Rule Test Completed Successfully! ---")
    print("Review console output for detailed diagnostic messages from the model's forward pass regarding spikes and delta_W.")

if __name__ == "__main__":
    run_new_plasticity_tests()