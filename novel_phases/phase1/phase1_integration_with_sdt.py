"""
Spiking Neural Network Layers for Decision Transformer
Integrates Adaptive Spiking Windows (Phase 1) with core SNN components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List, Union
import matplotlib.pyplot as plt


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with adaptive threshold
    Enhanced version with configurable parameters and better state management
    """
    def __init__(self, 
                 tau_mem: float = 20.0, 
                 tau_syn: float = 5.0, 
                 v_threshold: float = 1.0, 
                 v_reset: float = 0.0,
                 learnable_params: bool = True):
        super().__init__()
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn  
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        
        # Learnable decay parameters (can be disabled for faster inference)
        if learnable_params:
            self.beta = nn.Parameter(torch.tensor(np.exp(-1/tau_mem)))
            self.alpha = nn.Parameter(torch.tensor(np.exp(-1/tau_syn)))
        else:
            self.register_buffer('beta', torch.tensor(np.exp(-1/tau_mem)))
            self.register_buffer('alpha', torch.tensor(np.exp(-1/tau_syn)))
        
    def forward(self, x: torch.Tensor, state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass of LIF neuron
        Args:
            x: Input tensor [batch_size, ..., hidden_dim]
            state: Previous (v_mem, i_syn) state
        Returns:
            spikes: Binary spike tensor
            new_state: Updated (v_mem, i_syn) state
        """
        if state is None:
            v_mem = torch.zeros_like(x)
            i_syn = torch.zeros_like(x)
        else:
            v_mem, i_syn = state
            
        # Synaptic current update with decay
        i_syn = self.alpha * i_syn + x
        
        # Membrane potential update with leak
        v_mem = self.beta * v_mem + i_syn
        
        # Spike generation (threshold crossing)
        spikes = (v_mem >= self.v_threshold).float()
        
        # Reset membrane potential where spikes occurred
        v_mem = v_mem * (1 - spikes) + self.v_reset * spikes
        
        return spikes, (v_mem, i_syn)

    def reset_state(self, batch_size: int, *dims, device=None, dtype=None):
        """Reset neuron state for new sequence"""
        shape = (batch_size,) + dims
        v_mem = torch.zeros(shape, device=device, dtype=dtype)
        i_syn = torch.zeros(shape, device=device, dtype=dtype)
        return (v_mem, i_syn)


class SpikingLinear(nn.Module):
    """
    Linear layer with integrated spiking neurons
    Combines weight transformation with LIF dynamics
    """
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, spiking: bool = True, **lif_kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.spiking = spiking
        if spiking:
            self.lif = LIFNeuron(**lif_kwargs)
        
    def forward(self, x: torch.Tensor, state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """Forward pass with optional spiking"""
        x = self.linear(x)
        if self.spiking:
            return self.lif(x, state)
        else:
            return x, None


class AdaptiveSpikingAttention(nn.Module):
    """
    Spiking Self-Attention with Adaptive Temporal Windows
    Enhanced version of Phase 1 implementation with better integration
    """
    
    def __init__(self, embedding_dim: int, num_heads: int = 8, T_max: int = 20, 
                 lambda_reg: float = 1e-3, dropout: float = 0.1, 
                 learnable_lif: bool = True, complexity_weighting: float = 0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.T_max = T_max
        self.lambda_reg = lambda_reg
        self.complexity_weighting = complexity_weighting
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        # Standard attention projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)  
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # LIF neurons for Q, K, V with configurable learning
        self.lif_q = LIFNeuron(learnable_params=learnable_lif)
        self.lif_k = LIFNeuron(learnable_params=learnable_lif)
        self.lif_v = LIFNeuron(learnable_params=learnable_lif)
        
        # Enhanced adaptive window gating network
        self.window_gate = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Complexity estimator with better architecture
        self.complexity_estimator = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.head_dim) ** -0.5
        
        # Enhanced logging and analysis
        self.register_buffer('step_count', torch.tensor(0))
        self.T_history = []
        self.attention_entropy_history = []
        
    def get_adaptive_windows(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive time windows T_i for each token with enhanced logic
        """
        batch_size, seq_len, _ = x.shape
        
        # Base gating decision
        gate = self.window_gate(x)  # [batch_size, seq_len, 1]
        
        # Complexity-based adjustment
        complexity = self.complexity_estimator(x)  # [batch_size, seq_len, 1]
        
        # Enhanced combination with position bias
        position_bias = torch.linspace(0.8, 1.2, seq_len, device=x.device).view(1, -1, 1)
        
        # Combine all factors
        combined_score = (
            (1 - self.complexity_weighting) * gate + 
            self.complexity_weighting * complexity
        ) * position_bias
        
        # Convert to discrete time steps with smoother distribution
        T_i = torch.ceil(combined_score.squeeze(-1) * self.T_max).clamp(min=1, max=self.T_max).int()
        
        return T_i
    
    def generate_adaptive_spikes_vectorized(self, x: torch.Tensor, T_i: torch.Tensor, 
                                          lif_neuron: LIFNeuron) -> torch.Tensor:
        """
        Vectorized spike generation (more efficient than nested loops)
        """
        batch_size, seq_len, embedding_dim = x.shape
        device = x.device
        
        # Initialize output tensor
        all_spikes = torch.zeros(batch_size, seq_len, self.T_max, embedding_dim, 
                                device=device, dtype=x.dtype)
        
        # Create time mask for efficient processing
        time_mask = torch.arange(self.T_max, device=device).view(1, 1, -1) < T_i.unsqueeze(-1)
        
        # Process each time step
        state = lif_neuron.reset_state(batch_size, seq_len, embedding_dim, 
                                      device=device, dtype=x.dtype)
        
        for t in range(self.T_max):
            # Only process tokens that need this time step
            active_mask = time_mask[:, :, t]  # [batch_size, seq_len]
            
            if active_mask.any():
                # Expand input for active tokens
                x_t = x * active_mask.unsqueeze(-1).float()
                spikes, state = lif_neuron(x_t, state)
                all_spikes[:, :, t] = spikes * active_mask.unsqueeze(-1).float()
        
        return all_spikes
    
    def masked_attention_accumulation_efficient(self, q_spikes: torch.Tensor, k_spikes: torch.Tensor, 
                                              v_spikes: torch.Tensor, T_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficient attention computation using einsum operations
        """
        batch_size, seq_len, T_max, embedding_dim = q_spikes.shape
        
        # Reshape for multi-head attention
        q_spikes = q_spikes.view(batch_size, seq_len, T_max, self.num_heads, self.head_dim)
        k_spikes = k_spikes.view(batch_size, seq_len, T_max, self.num_heads, self.head_dim)  
        v_spikes = v_spikes.view(batch_size, seq_len, T_max, self.num_heads, self.head_dim)
        
        # Create temporal mask
        time_mask = torch.arange(T_max, device=T_i.device).view(1, 1, -1) < T_i.unsqueeze(-1)
        time_mask = time_mask.float().unsqueeze(-1).unsqueeze(-1)  # [B, S, T, 1, 1]
        
        # Apply temporal masking
        q_masked = q_spikes * time_mask
        k_masked = k_spikes * time_mask
        v_masked = v_spikes * time_mask
        
        # Efficient attention score computation using einsum
        # Sum over time dimension first, then compute attention
        q_sum = torch.sum(q_masked, dim=2)  # [B, S, H, D_h]
        k_sum = torch.sum(k_masked, dim=2)  # [B, S, H, D_h]
        v_mean = torch.mean(v_masked, dim=2)  # [B, S, H, D_h]
        
        # Compute attention scores: [B, H, S, S]
        attention_scores = torch.einsum('bihd,bjhd->bhij', q_sum, k_sum) * self.scale
        
        # Apply softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        v_mean = v_mean.transpose(1, 2)  # [B, H, S, D_h]
        out = torch.matmul(attention_weights, v_mean)  # [B, H, S, D_h]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim)
        
        return self.out_proj(out), attention_weights
    
    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute attention entropy for analysis"""
        # attention_weights: [B, H, S, S]
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-12), dim=-1)
        return entropy.mean()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """Enhanced forward pass with better metrics and efficiency"""
        batch_size, seq_len, embedding_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x) 
        v = self.v_proj(x)
        
        # Get adaptive time windows
        T_i = self.get_adaptive_windows(x)
        
        # Generate adaptive spikes (using vectorized version)
        q_spikes = self.generate_adaptive_spikes_vectorized(q, T_i, self.lif_q)
        k_spikes = self.generate_adaptive_spikes_vectorized(k, T_i, self.lif_k)
        v_spikes = self.generate_adaptive_spikes_vectorized(v, T_i, self.lif_v)
        
        # Compute attention with efficient accumulation
        output, attention_weights = self.masked_attention_accumulation_efficient(
            q_spikes, k_spikes, v_spikes, T_i
        )
        
        # Compute losses and metrics
        reg_loss = self.compute_regularization_loss(T_i)
        attention_entropy = self.compute_attention_entropy(attention_weights)
        
        # Enhanced logging
        if self.training:
            self.T_history.append(T_i.cpu().numpy())
            self.attention_entropy_history.append(attention_entropy.item())
            self.step_count += 1
        
        # Comprehensive metrics
        metrics = {
            'T_mean': T_i.float().mean().item(),
            'T_std': T_i.float().std().item(),
            'T_min': T_i.min().item(),
            'T_max': T_i.max().item(),
            'T_efficiency': (T_i.float().mean() / self.T_max).item(),
            'reg_loss': reg_loss.item(),
            'attention_entropy': attention_entropy.item(),
            'attention_weights': attention_weights.detach(),
            'adaptive_windows': T_i.detach(),
            'sparsity': (T_i < self.T_max).float().mean().item()
        }
        
        return output, metrics
    
    def compute_regularization_loss(self, T_i: torch.Tensor) -> torch.Tensor:
        """Enhanced regularization with variance penalty"""
        T_mean = T_i.float().mean()
        T_var = T_i.float().var()
        return self.lambda_reg * (T_mean + 0.1 * T_var)  # Encourage both efficiency and consistency


class SpikingTransformerBlock(nn.Module):
    """
    Complete transformer block with adaptive spiking attention
    Integrates seamlessly with standard transformer architectures
    """
    def __init__(self, embedding_dim: int, num_heads: int = 8, T_max: int = 20,
                 ff_dim: Optional[int] = None, dropout: float = 0.1, **attention_kwargs):
        super().__init__()
        
        ff_dim = ff_dim or 4 * embedding_dim
        
        # Adaptive spiking attention
        self.attention = AdaptiveSpikingAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            T_max=T_max,
            dropout=dropout,
            **attention_kwargs
        )
        
        # Standard components
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Feed-forward with optional spiking
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """Standard transformer block forward with residual connections"""
        # Self-attention with residual connection
        attn_out, metrics = self.attention(self.norm1(x), mask)
        x = x + attn_out
        
        # Feed-forward with residual connection
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        
        return x, metrics


class SpikingDecisionTransformer(nn.Module):
    """
    Decision Transformer with Adaptive Spiking Windows
    Complete integration for RL applications
    """
    def __init__(self, 
                 state_dim: int,
                 action_dim: int, 
                 embedding_dim: int = 128,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 T_max: int = 20,
                 max_length: int = 20, # Context window K for transformer
                 max_episode_len: int = 1000, # Max steps in an episode for timestep embedding
                 dropout: float = 0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.max_length = max_length # K, context window
        self.max_episode_len = max_episode_len
        
        # Embeddings
        self.state_embedding = nn.Linear(state_dim, embedding_dim)
        self.action_embedding = nn.Linear(action_dim, embedding_dim)
        self.return_embedding = nn.Linear(1, embedding_dim)
        self.timestep_embedding = nn.Embedding(self.max_episode_len, embedding_dim) # Use max_episode_len here
        
        # Positional embeddings
        # max_length here is K (context window). The sequence given to transformer is K*3 tokens long.
        self.position_embedding = nn.Embedding(max_length * 3, embedding_dim)
        
        # Spiking transformer layers
        self.layers = nn.ModuleList([
            SpikingTransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                T_max=T_max,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output heads
        self.norm = nn.LayerNorm(embedding_dim)
        self.action_head = nn.Linear(embedding_dim, action_dim)
        
        # For metrics aggregation
        self.layer_metrics = []
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor, 
                returns_to_go: torch.Tensor, timesteps: torch.Tensor) -> Dict:
        """
        Forward pass for decision transformer with spiking attention
        
        Args:
            states: [batch_size, seq_len, state_dim]
            actions: [batch_size, seq_len, action_dim]  
            returns_to_go: [batch_size, seq_len, 1]
            timesteps: [batch_size, seq_len]
        """
        batch_size, seq_len = states.shape[:2]
        
        # Embed all inputs
        state_embs = self.state_embedding(states)
        action_embs = self.action_embedding(actions)
        return_embs = self.return_embedding(returns_to_go)
        
        # Stack embeddings: [R_t, s_t, a_t] form a triplet for each time step t
        # Sequence becomes: [R_0, s_0, a_0,  R_1, s_1, a_1, ... R_{K-1}, s_{K-1}, a_{K-1}]
        # Resulting in a sequence of length K*3 for the transformer
        stacked_embs = torch.stack(
            [return_embs, state_embs, action_embs], dim=2
        ) # Shape: [batch_size, seq_len, 3, embedding_dim]
        
        sequence_embeddings = stacked_embs.reshape(
            batch_size, seq_len * 3, self.embedding_dim
        ) # Shape: [batch_size, seq_len * 3, embedding_dim]
        
        # Add time embeddings
        # `timesteps` are [batch_size, seq_len], representing the time index of (s_t) or the triplet (R_t,s_t,a_t)
        # `self.timestep_embedding` expects indices up to `self.max_length` (which is K)
        time_embs = self.timestep_embedding(timesteps) # Shape: [batch_size, seq_len, embedding_dim]
        # Each time embedding corresponds to a (R,s,a) group, so repeat it for each element in the group
        time_embs_repeated = time_embs.repeat_interleave(3, dim=1) # Shape: [batch_size, seq_len * 3, embedding_dim]
        sequence_embeddings = sequence_embeddings + time_embs_repeated
        
        # Add positional embeddings
        # The sequence length for position embeddings is seq_len * 3
        positions = torch.arange(3 * seq_len, device=states.device)
        sequence_embeddings = sequence_embeddings + self.position_embedding(positions)
        
        # Pass through spiking transformer layers
        x = sequence_embeddings
        all_metrics = []
        
        for layer in self.layers:
            x, metrics = layer(x)
            all_metrics.append(metrics)
        
        x = self.norm(x)
        
        # Extract action predictions (every 3rd token starting from index 2)
        action_tokens = x[:, 2::3]  # [batch_size, seq_len, embedding_dim]
        action_predictions = self.action_head(action_tokens)
        
        # Aggregate metrics across layers
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        
        return {
            'action_predictions': action_predictions,
            'metrics': aggregated_metrics,
            'embeddings': x
        }
    
    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across all layers"""
        if not all_metrics:
            return {}
        
        aggregated = {}
        for key in all_metrics[0].keys():
            if key in ['attention_weights', 'adaptive_windows']:
                # Keep per-layer for detailed analysis
                aggregated[f'layer_{key}'] = [m[key] for m in all_metrics]
            else:
                # Average across layers
                values = [m[key] for m in all_metrics if isinstance(m[key], (int, float))]
                if values:
                    aggregated[f'avg_{key}'] = sum(values) / len(values)
                    aggregated[f'layer_{key}'] = values
        
        return aggregated


# Utility functions for integration

def get_default_config():
    """Get default configuration for spiking decision transformer"""
    return {
        'state_dim': 17,  # Example for continuous control
        'action_dim': 6,  # Example for continuous control
        'embedding_dim': 128,
        'num_layers': 6,
        'num_heads': 8,
        'T_max': 20,
        'max_length': 20, # K, context window
        'max_episode_len': 1000, # Default max episode length for timestep embedding
        'dropout': 0.1,
        'lambda_reg': 1e-3
    }

def validate_config(config: dict) -> dict:
    """Validate and complete configuration"""
    config = config.copy()
    required_fields = ['state_dim', 'action_dim']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field '{field}' missing from config")
    defaults = get_default_config()
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
    assert config['embedding_dim'] % config['num_heads'] == 0, \
        "embedding_dim must be divisible by num_heads"
    assert config['T_max'] > 0, "T_max must be positive"
    assert 0 <= config['dropout'] <= 1, "dropout must be in [0, 1]"
    return config

def create_spiking_dt_model(config: Dict) -> SpikingDecisionTransformer:
    """Factory function to create spiking DT model from config"""
    return SpikingDecisionTransformer(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        embedding_dim=config.get('embedding_dim', 128),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        T_max=config.get('T_max', 20),
        max_length=config.get('max_length', 20), # K
        max_episode_len=config.get('max_episode_len', 1000), # Max episode len
        dropout=config.get('dropout', 0.1)
    )

def compute_spiking_loss(action_loss: torch.Tensor, 
                        metrics: Dict, reg_weight: float = 1.0) -> torch.Tensor:
    """Compute total loss including spiking regularization and a pre-computed action_loss."""
    # Regularization from spiking attention
    model_reg_loss = 0.0
    if 'avg_reg_loss' in metrics:
        model_reg_loss = metrics['avg_reg_loss'] # Use the value from metrics
    
    total_loss = action_loss + reg_weight * model_reg_loss
    
    return total_loss