import torch
import sys
import os

# Add snn-dt to the python path to allow imports from src
sys.path.insert(0, os.path.abspath("snn-dt"))

from src.models.snn_dt import SnnDt

# A simplified MockConfig, similar to the one in the tests
class MockConfig:
    def __init__(self):
        self.model = self.Model()
        self.dataset = self.Dataset()
        self.snn = self.Snn()
        self.env = "dummy_env"

    class Model:
        name = "snn_dt"
        d_model = 128
        n_heads = 4
        n_layers = 2
        
    class Dataset:
        state_dim = 4
        act_dim = 1
        max_timesteps = 100
        is_discrete = False
        
    class Snn:
        lif_tau = 20.0
        surrogate_k = 25.0
        use_plasticity = False

    class Training:
        device = "cpu"

def debug_snn_dt():
    """
    Instantiates the SnnDt model, runs a forward pass, and checks the spike count.
    """
    print("--- Initializing SNN-DT Debug Script ---")
    
    # 1. Setup model and config
    cfg = MockConfig()
    model = SnnDt(cfg)
    model.eval()  # Use eval mode to disable training-specific logic like plasticity

    print("SnnDt model instantiated successfully.")

    # 2. Create a batch of dummy data with high magnitude to encourage spiking
    batch = {
        "states": torch.randn(16, 20, 4) * 100,
        "actions": torch.randn(16, 20, 1) * 100,
        "returns_to_go": torch.randn(16, 20, 1) * 100,
        "timesteps": torch.randint(0, 100, (16, 20)),
        "mask": torch.ones(16, 20),
    }

    print("Batch created. Running a single forward pass...")

    # 3. Run the forward pass and check spikes
    with torch.no_grad():
        model(batch)
    
    spike_count_1 = model.count_spikes()
    print(f"Spike count after first pass: {spike_count_1}")

    # 4. Run a second pass to check accumulation
    print("Running a second forward pass to check accumulation...")
    with torch.no_grad():
        model(batch)

    spike_count_2 = model.count_spikes()
    print(f"Spike count after second pass: {spike_count_2}")
    
    # 5. Check the reset mechanism
    print("Resetting spike counts...")
    model.reset_spike_counts()
    print(f"Spike count after reset: {model.count_spikes()}")
    
    print("--- Debug Script Finished ---")

if __name__ == "__main__":
    debug_snn_dt()