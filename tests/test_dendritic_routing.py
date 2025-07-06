import pytest
import torch
import torch.nn.functional as F
# Adjust import path as needed
from novel_phases.phase3.dendritic_routing import DendriticRouter

@pytest.fixture
def dr_model():
    """Fixture for DendriticRouter model."""
    num_heads = 4
    return DendriticRouter(num_heads=num_heads)

def test_dr_initialization(dr_model):
    """Test if the DendriticRouter initializes correctly."""
    assert dr_model.H == 4, "Number of heads not initialized correctly"
    assert isinstance(dr_model.routing_mlp, torch.nn.Sequential), "routing_mlp should be nn.Sequential"
    assert len(dr_model.routing_mlp) == 2, "routing_mlp should have two layers (Linear, Sigmoid)"
    assert isinstance(dr_model.routing_mlp[0], torch.nn.Linear), "First layer of MLP should be Linear"
    assert dr_model.routing_mlp[0].in_features == 4, "MLP Linear layer input features mismatch"
    assert dr_model.routing_mlp[0].out_features == 4, "MLP Linear layer output features mismatch"
    assert isinstance(dr_model.routing_mlp[1], torch.nn.Sigmoid), "Second layer of MLP should be Sigmoid"

def test_dr_forward_pass_output_shape(dr_model):
    """Test the output shape of the forward pass."""
    B, L, H, d = 2, 5, dr_model.H, 8  # Batch, SeqLen, NumHeads, FeatureDim
    # y_heads assumed to be summed over T already: [B, L, H, d]
    y_heads_input = torch.randn(B, L, H, d)
    
    gated_output = dr_model(y_heads_input)
    
    # Expected output shape: [B, L, d] (after summing over H dimension)
    assert gated_output.shape == (B, L, d), \
        f"Output shape mismatch. Expected ({B}, {L}, {d}), got {gated_output.shape}"

def test_dr_gates_values(dr_model):
    """Test if the computed gates are between 0 and 1 (due to Sigmoid)."""
    B, L, H, d = 2, 5, dr_model.H, 8
    y_heads_input = torch.randn(B, L, H, d)

    # To inspect gates, we can hook into the forward pass or call parts of it
    summary = y_heads_input.sum(dim=-1)  # [B, L, H]
    gates = dr_model.routing_mlp(summary.view(-1, H)) # [(B*L), H]
    gates = gates.view(B, L, H) # [B, L, H]

    assert torch.all(gates >= 0) and torch.all(gates <= 1), \
        "Gate values should be between 0 and 1 due to Sigmoid."

def test_dr_forward_pass_device_consistency(dr_model):
    """Test if the model handles inputs and produces outputs on the same device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dr_model.to(device)
        
        B, L, H, d = 2, 5, dr_model.H, 8
        y_heads_input = torch.randn(B, L, H, d).to(device)
        
        gated_output = dr_model(y_heads_input)
        
        assert gated_output.device == device, "Output tensor not on the correct device"
        for param in dr_model.parameters():
            assert param.device == device, "Model parameter not on the correct device"
    else:
        pytest.skip("CUDA not available for device test")

def test_dr_routing_mlp_non_trivial_weights_after_training_placeholder(dr_model):
    """
    Placeholder test for checking non-trivial gating patterns.
    This would require a minimal training loop to see if weights diverge.
    """
    initial_weights = dr_model.routing_mlp[0].weight.clone().detach()

    # Dummy training loop
    optimizer = torch.optim.SGD(dr_model.parameters(), lr=0.01)
    B, L, H, d = 2, 5, dr_model.H, 8
    y_heads_input = torch.randn(B, L, H, d)
    # Artificial target to make weights learn something
    dummy_target = torch.randn(B, L, d) 

    for _ in range(5): # Few steps
        optimizer.zero_grad()
        gated_output = dr_model(y_heads_input)
        loss = F.mse_loss(gated_output, dummy_target)
        loss.backward()
        optimizer.step()

    final_weights = dr_model.routing_mlp[0].weight.detach()
    
    assert not torch.allclose(initial_weights, final_weights, atol=1e-5), \
        "MLP weights should have changed after dummy training."
    # A more sophisticated check might involve looking at the distribution of weights
    # or their magnitude, but simple change is a first step.
    print(f"\nInitial MLP Weights (sample): {initial_weights[0,:2].numpy().round(3)}")
    print(f"Final MLP Weights (sample):   {final_weights[0,:2].numpy().round(3)}")

# To run: (similar to the previous test file)
# Ensure pytest, torch are installed.
# Ensure __init__.py files are in src/ and src/models/.
# Run from project root: pytest SpikingMindRL/tests/test_dendritic_routing.py
# Or (if project root is parent of SpikingMindRL): pytest
# PYTHONPATH=. pytest SpikingMindRL/tests/test_dendritic_routing.py (if needed)