# src/adapters/sdt_adapter.py
"""
Robust adapter to discover and load the local SpikingDecisionTransformer class by scanning
likely source files under src/ and src/models/. This avoids relative-import issues
when running scripts with `python -m`.
"""

from pathlib import Path
import importlib.util
import sys
import time
import torch
import traceback

# repo root relative to this file (src/adapters -> parents[2] is repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
CAND_DIRS = [SRC_ROOT, SRC_ROOT / "models"]

# candidate class names to look for
CLASS_CANDIDATES = [
    "SpikingDecisionTransformer",
    "SpikingDT",
    "SpikingDecisionTransformerModel",
    "SpikingDecisionTransformerModule",
    "SDT",
    "SpikingTransformer",
]

def _scan_and_find_class():
    """Scan candidate directories for python files, try to import them (by path),
    and search for one of the candidate class names."""
    scanned = []
    import_errors = {}
    for d in CAND_DIRS:
        if not d.exists():
            continue
        for py in sorted(d.glob("*.py")):
            scanned.append(py)
            try:
                spec = importlib.util.spec_from_file_location(f"local_src_{py.stem}", str(py))
                mod = importlib.util.module_from_spec(spec)
                loader = spec.loader
                if loader is None:
                    import_errors[str(py)] = "no loader"
                    continue
                try:
                    loader.exec_module(mod)  # execute file (may import dependencies)
                except Exception as e_mod:
                    # store traceback and continue scanning
                    import_errors[str(py)] = traceback.format_exc()
                    continue
                # search for any candidate class
                for cname in CLASS_CANDIDATES:
                    if hasattr(mod, cname):
                        return getattr(mod, cname), str(py)
            except Exception as e:
                import_errors[str(py)] = traceback.format_exc()
                continue
    # If not found, raise informative ImportError
    msg_lines = [
        "Could not find a SpikingDecisionTransformer class in the scanned src files.",
        "Scanned files:",
    ]
    for p in scanned:
        msg_lines.append(f"  - {p}")
    if import_errors:
        msg_lines.append("\nImport errors encountered while scanning (showing filenames with exceptions):")
        for fname, tb in import_errors.items():
            msg_lines.append(f"--- {fname} ---")
            # keep brief: only first 10 lines of traceback
            tb_lines = tb.splitlines()
            msg_lines.extend(tb_lines[:10])
            if len(tb_lines) > 10:
                msg_lines.append("  (truncated traceback...)")
    msg_lines.append("\nIf your SDT class uses a different name or is nested, update CLASS_CANDIDATES "
                     "or move the model file to src/ or src/models/.")
    raise ImportError("\n".join(msg_lines))


# find model class dynamically
ModelClass, model_file = _scan_and_find_class()


class SDTWrapper:
    def __init__(self, config: dict, device="cpu"):
        """
        config: dict passed to the discovered model constructor.
                If the constructor requires different args, inspect the file printed below:
                {model_file}
        """
        self.device = device
        try:
            self.model = ModelClass(**config)
        except TypeError as e:
            raise TypeError(
                "Could not instantiate the detected SDT class with the provided config dict.\n"
                f"Constructor TypeError: {e}\n"
                f"Inspect the file: {model_file} for the expected signature and adapt your config."
            ) from e
        except Exception as e:
            raise RuntimeError(
                "An error occurred while constructing the SDT model.\n"
                f"Original error: {e}"
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
            # try common API methods
            if hasattr(self.model, "act"):
                out = self.model.act(obs_t)
            elif hasattr(self.model, "forward"):
                out = self.model.forward(obs_t)
            elif hasattr(self.model, "__call__"):
                out = self.model(obs_t)
            else:
                raise AttributeError(
                    "SDT model has no callable inference method (act/forward/__call__). Inspect the model."
                )
            # if on GPU, make sure to sync if measuring times on GPU (user may modify)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

        action = out
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        try:
            action = action.squeeze()
        except Exception:
            pass
        return action, latency_ms
