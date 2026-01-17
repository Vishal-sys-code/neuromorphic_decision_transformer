import sys
import inspect
from pathlib import Path

# Add possible paths
root = Path('.').resolve()
sys.path.append(str(root))
sys.path.append(str(root / 'snn-dt')) # Try adding this to see if it fixes imports

def try_import(name, path):
    try:
        if name == 'cql':
            from src.models.cql import CQL as cls
        elif name == 'dt':
            from src.models.dt import DecisionTransformer as cls
        elif name == 'snn_dt':
            from src.models.snn_dt import SnnDt as cls
        elif name == 'dsformer':
            from src.models.dsformer import DsFormer as cls
        elif name == 'iql':
            from src.models.iql import IQL as cls
        
        print(f"[{name}] IMPORT SUCCESS")
        sig = inspect.signature(cls.__init__)
        print(f"[{name}] Init Signature: {sig}")
    except Exception as e:
        print(f"[{name}] IMPORT FAILED: {e}")

models = ['cql', 'dt', 'snn_dt', 'dsformer', 'iql']
for m in models:
    try_import(m, 'src.models')
