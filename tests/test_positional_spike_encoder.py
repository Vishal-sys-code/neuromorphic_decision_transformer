import pytest
import torch
# Adjust the import path based on your project structure and how pytest discovers tests
# If 'SpikingMindRL' is the root and in PYTHONPATH, and 'src' is a package:
from novel_phases.phase3.positional_spike_encoder import PositionalSpikeEncoder

# Or, if tests are run from within 'SpikingMindRL/tests' and 'src' is a sibling:
# from ..src.models.positional_spike_encoder import PositionalSpikeEncoder

# To make imports work smoothly, ensure __init__.py files exist in:
# SpikingMindRL/src/
# SpikingMindRL/src/models/
# SpikingMindRL/tests/ (optional, but good practice)

@pytest.fixture
def pse_model():
    """Fixture for PositionalSpikeEncoder model."""
    num_heads = 4
    window_length = 10
    return PositionalSpikeEncoder(num_heads=num_heads, window_length=window_length)

def test_pse_initialization(pse_model):
    """Test if the PositionalSpikeEncoder initializes correctly."""
    assert pse_model.H == 4, "Number of heads not initialized correctly"
    assert pse_model.T == 10, "Window length not initialized correctly"
    assert pse_model.freq.shape == (4,), "Frequency parameter shape is incorrect"
    assert pse_model.phase.shape == (4,), "Phase parameter shape is incorrect"
    assert pse_model.freq.requires_grad, "Frequency should be a learnable parameter"
    assert pse_model.phase.requires_grad, "Phase should be a learnable parameter"

def test_pse_forward_pass_output_shape(pse_model):
    """Test the output shape of the forward pass."""
    B, L, d = 2, 5, 8  # Batch size, Sequence length, Embedding dimension
    embeddings = torch.randn(B, L, d)
    
    pos_mask = pse_model(embeddings)
    
    # Expected shape of pos_mask is [H, T]
    assert pos_mask.shape == (pse_model.H, pse_model.T), \
        f"Output shape mismatch. Expected ({pse_model.H}, {pse_model.T}), got {pos_mask.shape}"

def test_pse_forward_pass_output_values(pse_model):
    """Test if the output values are binary (0 or 1)."""
    B, L, d = 2, 5, 8
    embeddings = torch.randn(B, L, d)
    
    pos_mask = pse_model(embeddings)
    
    assert torch.all((pos_mask == 0) | (pos_mask == 1)), \
        "Output mask should contain only binary values (0 or 1)."

def test_pse_parameters_device(pse_model):
    """Test if parameters are on the correct device (e.g., after moving the model)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        pse_model.to(device)
        
        B, L, d = 2, 5, 8
        embeddings = torch.randn(B, L, d).to(device)
        
        pos_mask = pse_model(embeddings)
        
        assert pse_model.freq.device == device, "Frequency parameter not on correct device"
        assert pse_model.phase.device == device, "Phase parameter not on correct device"
        assert pos_mask.device == device, "Output tensor not on correct device"
    else:
        pytest.skip("CUDA not available for device test")

# Example of how you might test for learned frequencies/phases after dummy training
# This would typically be an integration test rather than a unit test for the encoder itself.
# def test_pse_parameter_learning_divergence(pse_model):
#     """
#     Placeholder test to illustrate checking if parameters diverge from random.
#     This requires a minimal training loop.
#     """
#     initial_freq = pse_model.freq.clone().detach()
#     initial_phase = pse_model.phase.clone().detach()
#
#     # Dummy training loop
#     optimizer = torch.optim.SGD(pse_model.parameters(), lr=0.1)
#     B, L, d = 2, 5, 8
#     embeddings = torch.randn(B, L, d)
#     # Dummy target related to the mask, e.g., try to make the mask sum to a certain value
#     # This is highly artificial, real loss depends on the downstream task
#     dummy_target = torch.ones_like(pse_model(embeddings)) * 0.5 
#
#     for _ in range(5): # Few steps
#         optimizer.zero_grad()
#         pos_mask = pse_model(embeddings)
#         loss = torch.mse_loss(pos_mask, dummy_target) # Artificial loss
#         loss.backward()
#         optimizer.step()
#
#     final_freq = pse_model.freq.detach()
#     final_phase = pse_model.phase.detach()
#
#     assert not torch.allclose(initial_freq, final_freq), "Frequencies should have changed after training."
#     assert not torch.allclose(initial_phase, final_phase), "Phases should have changed after training."
#     print(f"\nInitial Freq: {initial_freq.numpy().round(3)}")
#     print(f"Final Freq:   {final_freq.numpy().round(3)}")
#     print(f"Initial Phase: {initial_phase.numpy().round(3)}")
#     print(f"Final Phase:   {final_phase.numpy().round(3)}")

# To run these tests:
# 1. Ensure you have pytest and torch installed: pip install pytest torch
# 2. Create __init__.py in SpikingMindRL/src and SpikingMindRL/src/models
#    touch SpikingMindRL/src/__init__.py
#    touch SpikingMindRL/src/models/__init__.py
# 3. Navigate to the directory containing 'SpikingMindRL' (e.g., the parent of SpikingMindRL)
# 4. Run pytest: pytest
#    Or, if SpikingMindRL is your project root: pytest SpikingMindRL/tests/test_positional_spike_encoder.py
#
# If you encounter import errors, you might need to adjust PYTHONPATH or how pytest is invoked.
# For example, if 'SpikingMindRL' is your project root:
# PYTHONPATH=. pytest SpikingMindRL/tests/test_positional_spike_encoder.py
# Or ensure your project is installed in editable mode if it's set up as a package.
#
# The import `from src.models.positional_spike_encoder import PositionalSpikeEncoder` assumes
# that the directory *containing* `src` is in the Python path, or that you run pytest
# from the directory containing `src`.
# A common setup is to have a project root (e.g., "SpikingMindRL_Project")
# SpikingMindRL_Project/
#  |- src/
#  |  |- models/
#  |  |  |- __init__.py
#  |  |  |- positional_spike_encoder.py
#  |  |- __init__.py
#  |- tests/
#  |  |- __init__.py (optional)
#  |  |- test_positional_spike_encoder.py
#  |- pyproject.toml (or setup.py)
#
# If running pytest from "SpikingMindRL_Project", the import should work.
# If your structure is SpikingMindRL/src and SpikingMindRL/tests directly,
# and you run pytest from SpikingMindRL/, then `from src...` is also correct.
# The key is that `src` is seen as a package.

# Create __init__.py files to make src and models importable as packages
# This is a common practice for structuring Python projects.
# (Agent: I will create these __init__.py files in a separate step if they don't exist)
# For now, this test file assumes they will exist.