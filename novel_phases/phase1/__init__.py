"""
SNN-DT Models Package
Integrated Spiking Neural Networks for Decision Transformers
"""

from ...models.adaptive_attention import (
    LIFNeuron as Phase1LIFNeuron,
    AdaptiveSpikingAttention as Phase1AdaptiveAttention
)

from ...models.spiking_layers import (
    LIFNeuron,
    SpikingLinear,
    AdaptiveSpikingAttention,
    SpikingTransformerBlock,
    SpikingDecisionTransformer,
    create_spiking_dt_model,
    compute_spiking_loss
)

# Version and phase tracking
__version__ = "0.1.0"
__phase__ = "Phase 1: Adaptive Spiking Windows"

# Main exports for external use
__all__ = [
    # Core spiking components
    'LIFNeuron',
    'SpikingLinear',
    'AdaptiveSpikingAttention',
    'SpikingTransformerBlock',
    
    # Main model
    'SpikingDecisionTransformer',
    
    # Utilities
    'create_spiking_dt_model',
    'compute_spiking_loss',
    
    # Phase 1 components (for research/comparison)
    'Phase1LIFNeuron',
    'Phase1AdaptiveAttention',
    
    # Metadata
    '__version__',
    '__phase__'
]

# Default model configuration
DEFAULT_CONFIG = {
    'state_dim': 17,  # Example for continuous control
    'action_dim': 6,  # Example for continuous control
    'embedding_dim': 128,
    'num_layers': 6,
    'num_heads': 8,
    'T_max': 20,
    'max_length': 1000,
    'dropout': 0.1,
    'lambda_reg': 1e-3
}

def get_default_config():
    """Get default configuration for spiking decision transformer"""
    return DEFAULT_CONFIG.copy()

def validate_config(config: dict) -> dict:
    """Validate and complete configuration"""
    config = config.copy()
    
    # Required fields
    required_fields = ['state_dim', 'action_dim']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field '{field}' missing from config")
    
    # Set defaults for optional fields
    for key, default_value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = default_value
    
    # Validation checks
    assert config['embedding_dim'] % config['num_heads'] == 0, \
        "embedding_dim must be divisible by num_heads"
    assert config['T_max'] > 0, "T_max must be positive"
    assert 0 <= config['dropout'] <= 1, "dropout must be in [0, 1]"
    
    return config