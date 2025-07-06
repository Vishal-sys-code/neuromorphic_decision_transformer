"""Models package for SpikingMindRL."""

# Commented out until the module is implemented
# from .simple_decision_transformer import SimpleDecisionTransformer

# __all__ = [] # Removed SimpleDecisionTransformer from __all__ 
from .spiking_layers import SpikingSelfAttention

# This file makes 'models' a Python sub-package.
# You can also use it to control what is imported when someone does 'from .models import *'
# For example:
# from .snn_dt import SNNDT
# from .positional_spike_encoder import PositionalSpikeEncoder
# from .dendritic_routing import DendriticRouter

# Making them available for easier import if desired:
# __all__ = ['SNNDT', 'PositionalSpikeEncoder', 'DendriticRouter']