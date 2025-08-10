# src/adapters/dsformer_adapter.py
"""
Adapter to load DecisionSpikeFormer from the external submodule by file path.
This avoids relative-import problems when running as a module.
"""

from pathlib import Path
import importlib.util
import sys
import time
import torch

# locate repo root relative to this file: src/adapters/... -> parents[2] is repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DSF_ROOT = REPO_ROOT / "external" / "DecisionSpikeFormer"
MODEL_FILE = DSF_ROOT / "gym" / "models" / "decision_spikeformer_pssa.py"

if not MODEL_FILE.exists():
    raise ImportError(
        f"DecisionSpikeFormer model file not found at expected path:\n  {MODEL_FILE}\n"
        "Make sure you added the external/DecisionSpikeFormer submodule or copied the file there."
    )

# load module from file path (robust to package context)
_spec = importlib.util.spec_from_file_location("decision_spikeformer_pssa", str(MODEL_FILE))
_dsf_mod = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_dsf_mod)  # type: ignore
except Exception as e:
    raise ImportError(f"Failed to execute DecisionSpikeFormer module at {MODEL_FILE}.\n"
                      f"Original error: {e}") from e

# try to fetch class
if hasattr(_dsf_mod, "DecisionSpikeFormer"):
    DecisionSpikeFormer = getattr(_dsf_mod, "DecisionSpikeFormer")
else:
    # helpful fail message listing what's available
    available = [n for n in dir(_dsf_mod) if not n.startswith("_")]
    raise ImportError(
        f"Loaded module {MODEL_FILE} but didn't find class 'DecisionSpikeFormer'.\n"
        f"Available names in module: {available}\n"
        "Open the file and check the classname (or update this adapter)."
    )


class DSFWrapper:
    def __init__(self, config: dict, device="cpu"):
        """
        config: dict passed to the DecisionSpikeFormer constructor.
                You may need to adapt keys to match the DSF constructor signature.
        """
        self.device = device
        # instantiate model (may need to adapt call if DSF expects different args)
        try:
            self.model = DecisionSpikeFormer(**config)
        except TypeError as e:
            # give an informative error to help debugging constructor signature mismatch
            raise TypeError(
                "Could not instantiate DecisionSpikeFormer with the provided config dict.\n"
                f"Constructor TypeError: {e}\n"
                "Inspect external/DecisionSpikeFormer/gym/models/decision_spikeformer_pssa.py "
                "for the expected constructor signature and adjust the config you pass."
            ) from e

        self.model.to(device)
        self.model.eval()
        self.spike_count = 0
        self._register_spike_hooks()

    def _register_spike_hooks(self):
        """Register forward hooks on modules that look like spiking modules to count spikes."""
        for name, module in self.model.named_modules():
            cls_name = module.__class__.__name__.lower()
            if "lif" in cls_name or "spike" in cls_name or "spiking" in cls_name:
                module.register_forward_hook(self._make_hook())

    def _make_hook(self):
        def hook(module, inp, out):
            try:
                out_tensor = out if isinstance(out, torch.Tensor) else out[0]
                self.spike_count += int((out_tensor != 0).sum().item())
            except Exception:
                # keep counting robust to unusual output shapes
                pass
        return hook

    def reset_spike_count(self):
        self.spike_count = 0

    def step(self, observation):
        """
        observation: numpy array or torch tensor; shape must match model expectation.
        returns: (action_numpy, latency_ms)
        """
        import numpy as np
        self.model.eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            obs_t = (observation if isinstance(observation, torch.Tensor)
                     else torch.tensor(observation)).unsqueeze(0).float().to(self.device)
            # Try common API names in order
            if hasattr(self.model, "act"):
                out = self.model.act(obs_t)
            elif hasattr(self.model, "forward"):
                out = self.model.forward(obs_t)
            else:
                raise AttributeError(
                    "DSF model has neither 'act' nor 'forward' methods. Inspect the model API."
                )
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

        # normalize action output
        action = out
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        # squeeze batch dim if present
        if isinstance(action, (list, tuple)) or (hasattr(action, "shape") and len(action.shape) > 0):
            try:
                action = action.squeeze()
            except Exception:
                pass
        return action, latency_ms