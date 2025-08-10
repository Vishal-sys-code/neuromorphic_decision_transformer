# src/eval/spike_counter.py
import torch

def attach_spike_counter(model, keywords=('lif','spike')):
    counter = {'count':0}
    def make_hook(counter):
        def hook(mod, inp, out):
            try:
                out_t = out if isinstance(out, torch.Tensor) else out[0]
                counter['count'] += (out_t != 0).sum().item()
            except Exception:
                pass
        return hook
    hooks = []
    for n,m in model.named_modules():
        if any(k in m.__class__.__name__.lower() for k in keywords):
            hooks.append(m.register_forward_hook(make_hook(counter)))
    return counter, hooks
