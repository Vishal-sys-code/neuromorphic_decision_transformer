import numpy as np

# Monkey patch to fix deprecated np.bool8 alias warning
if hasattr(np, 'bool8'):
    np.bool8 = np.bool_