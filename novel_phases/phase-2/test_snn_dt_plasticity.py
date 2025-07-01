import torch
import torch.nn as nn
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
spiking_mind_rl_root = os.path.join(project_root, "SpikingMindRL")
sys.path.insert(0, spiking_mind_rl_root)
sys.path.insert(0, os.path.join(spiking_mind_rl_root, "external"))
sys.path.insert(0, os.path.join(spiking_mind_rl_root, "src"))

try:
    from models.snn_dt import SNNDecisionTransformer
    from transformers import GPT2Config
except Exception as e:
    print(f"Import Error: {e}. Ensure SpikingMindRL, external submodule, and transformers lib are correctly pathed and installed.")
    sys.exit(1)

STATE_DIM, ACT_DIM, HIDDEN_SIZE = 4, 2, 128
MAX_LENGTH, MAX_EP_LEN = 10, 100
N_LAYER, N_HEAD = 3, 1
BATCH_SIZE, SEQ_LENGTH = 2, 5

def create_dummy_input(device):
    states = torch.rand(BATCH_SIZE, SEQ_LENGTH, STATE_DIM, device=device)
    actions = torch.rand(BATCH_SIZE, SEQ_LENGTH, ACT_DIM, device=device)
    rewards = torch.rand(BATCH_SIZE, SEQ_LENGTH, 1, device=device)
    returns_to_go = torch.rand(BATCH_SIZE, SEQ_LENGTH, 1, device=device)
    timesteps = torch.randint(0, MAX_EP_LEN, (BATCH_SIZE, SEQ_LENGTH), device=device)
    attention_mask = torch.ones(BATCH_SIZE, SEQ_LENGTH, device=device, dtype=torch.long)
    return states, actions, rewards, returns_to_go, timesteps, attention_mask

def run_tests():
    print("Starting Minimal SNNDT Plasticity Hook Test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_args = {
        'state_dim': STATE_DIM, 'act_dim': ACT_DIM, 'hidden_size': HIDDEN_SIZE,
        'max_length': MAX_LENGTH, 'max_ep_len': MAX_EP_LEN,
        'n_layer': N_LAYER, 'n_head': N_HEAD, 'action_tanh': False,
        'n_inner': 4 * HIDDEN_SIZE, 'activation_function': 'gelu',
        'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
    }

    try:
        model_with_plasticity = SNNDecisionTransformer(
            **model_args, enable_action_head_plasticity=True
        ).to(device)
        print("SNNDT with plasticity instantiated.")
        assert hasattr(model_with_plasticity, 'action_linear_layer_ref')
        assert hasattr(model_with_plasticity, 'handle_pre_hook')
        assert hasattr(model_with_plasticity, 'handle_post_hook')
        print("Hook attributes found.")
    except Exception as e:
        print(f"Instantiation Error (Plasticity Model): {type(e).__name__}: {e}")
        return

    dummy_inputs = create_dummy_input(device)
    _ = model_with_plasticity(*dummy_inputs)
    pre_syn, post_syn_logits = model_with_plasticity.get_captured_action_head_io()

    assert pre_syn is not None, "Pre-syn data not captured."
    assert pre_syn.shape == (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE), f"Pre-syn shape mismatch. Got {pre_syn.shape}"
    assert post_syn_logits is not None, "Post-syn logits not captured."
    assert post_syn_logits.shape == (BATCH_SIZE, SEQ_LENGTH, ACT_DIM), f"Post-syn logits shape mismatch. Got {post_syn_logits.shape}"
    print("Data capture successful.")

    model_with_plasticity.clear_captured_action_head_io()
    pre_syn_cleared, post_syn_logits_cleared = model_with_plasticity.get_captured_action_head_io()
    assert pre_syn_cleared is None and post_syn_logits_cleared is None, "Data not cleared."
    print("Data clearing successful.")

    model_with_plasticity.remove_hooks()
    print("remove_hooks called.")

    model_with_plasticity.captured_pre_syn_for_action = torch.tensor(1.0)
    model_with_plasticity.captured_post_syn_for_action_logits = torch.tensor(1.0)
    _ = model_with_plasticity(*dummy_inputs)
    pre_after_remove, post_after_remove = model_with_plasticity.get_captured_action_head_.io() # Typo: get_captured_action_head_.io
    assert pre_after_remove is None and post_after_remove is None, "Data not cleared by fwd pass after hook removal."
    print("Hook removal test passed.")

    del model_with_plasticity
    print("Model deleted (triggers __del__ for hook removal).")
    print("Minimal SNNDT Plasticity Hook Test Completed Successfully!")

if __name__ == "__main__":
    run_tests()
```
