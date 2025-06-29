"""
Adaptive Spiking Windows Implementation
Phase 1: Token-wise Temporal Allocation for Spiking Transformers

This module implements adaptive spiking attention where each token
can choose its own temporal spike window based on semantic complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional, Dict, List

class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with adaptive threshold"""
    def __init__(self, tau_mem: float = 20.0, tau_syn: float = 5.0, 
                 v_threshold: float = 1.0, v_reset: float = 0.0):
        super().__init__()
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn  
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        
        # Learnable decay parameters
        self.beta = nn.Parameter(torch.tensor(np.exp(-1/tau_mem)))
        self.alpha = nn.Parameter(torch.tensor(np.exp(-1/tau_syn)))
        
    def forward(self, x: torch.Tensor, state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        x: [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
        Returns: (spikes, (v_mem, i_syn))
        """
        if state is None:
            v_mem = torch.zeros_like(x)
            i_syn = torch.zeros_like(x)
        else:
            v_mem, i_syn = state
            
        # Synaptic current update
        i_syn = self.alpha * i_syn + x
        
        # Membrane potential update  
        v_mem = self.beta * v_mem + i_syn
        
        # Spike generation
        spikes = (v_mem >= self.v_threshold).float()
        
        # Reset membrane potential where spikes occurred
        v_mem = v_mem * (1 - spikes) + self.v_reset * spikes
        
        return spikes, (v_mem, i_syn)

class AdaptiveSpikingAttention(nn.Module):
    """Spiking Self-Attention with Adaptive Temporal Windows"""
    
    def __init__(self, embedding_dim: int, num_heads: int = 8, T_max: int = 20, 
                 lambda_reg: float = 1e-3, dropout: float = 0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.T_max = T_max
        self.lambda_reg = lambda_reg
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        # Standard attention projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)  
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # LIF neurons for Q, K, V
        self.lif_q = LIFNeuron()
        self.lif_k = LIFNeuron()
        self.lif_v = LIFNeuron()
        
        # Adaptive window gating network
        self.window_gate = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Complexity estimator (helps gate make better decisions)
        self.complexity_estimator = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.head_dim) ** -0.5
        
        # For logging and analysis
        self.register_buffer('step_count', torch.tensor(0))
        self.T_history = []
        
    def get_adaptive_windows(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive time windows T_i for each token
        x: [batch_size, seq_len, embedding_dim]
        Returns: T_i [batch_size, seq_len] with values in [1, T_max]
        """
        batch_size, seq_len, _ = x.shape
        
        # Base gating decision
        gate = self.window_gate(x)  # [batch_size, seq_len, 1]
        
        # Optional: Add complexity-based adjustment
        complexity = self.complexity_estimator(x)  # [batch_size, seq_len, 1]
        
        # Combine gate and complexity (complexity boosts difficult tokens)
        combined_score = 0.7 * gate + 0.3 * complexity
        
        # Convert to discrete time steps [1, T_max]
        T_i = torch.ceil(combined_score.squeeze(-1) * self.T_max).clamp(min=1, max=self.T_max).int()
        
        return T_i
    
    def generate_adaptive_spikes(self, x: torch.Tensor, T_i: torch.Tensor, 
                               lif_neuron: LIFNeuron) -> torch.Tensor:
        """
        Generate spikes with adaptive time windows per token
        x: [batch_size, seq_len, embedding_dim] 
        T_i: [batch_size, seq_len] adaptive windows
        Returns: [batch_size, seq_len, T_max, embedding_dim] (padded)
        """
        batch_size, seq_len, embedding_dim = x.shape
        device = x.device
        
        # Initialize output tensor
        all_spikes = torch.zeros(batch_size, seq_len, self.T_max, embedding_dim, 
                                device=device, dtype=x.dtype)
        
        # Process each batch and token
        for b in range(batch_size):
            for i in range(seq_len):
                T_current = T_i[b, i].item()
                token_input = x[b, i:i+1]  # [1, embedding_dim]
                
                # Generate spikes for this token's time window
                state = None
                for t in range(T_current):
                    spikes, state = lif_neuron(token_input, state)
                    all_spikes[b, i, t] = spikes.squeeze(0)
        
        return all_spikes
    
    def masked_attention_accumulation(self, q_spikes: torch.Tensor, k_spikes: torch.Tensor, 
                                    v_spikes: torch.Tensor, T_i: torch.Tensor) -> torch.Tensor:
        """
        Compute attention with time-masked spike accumulation
        """
        batch_size, seq_len, T_max, embedding_dim = q_spikes.shape
        
        # Reshape for multi-head attention
        q_spikes = q_spikes.view(batch_size, seq_len, T_max, self.num_heads, self.head_dim)
        k_spikes = k_spikes.view(batch_size, seq_len, T_max, self.num_heads, self.head_dim)  
        v_spikes = v_spikes.view(batch_size, seq_len, T_max, self.num_heads, self.head_dim)
        
        # Initialize attention scores
        attention_scores = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, 
                                     device=q_spikes.device, dtype=q_spikes.dtype)
        
        # Accumulate attention scores with temporal masking
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    T_i_val = T_i[b, i].item()
                    T_j_val = T_i[b, j].item()
                    
                    # Accumulate over valid time steps
                    valid_steps = min(T_i_val, T_j_val)
                    if valid_steps > 0:
                        # [T_steps, num_heads, head_dim] @ [T_steps, num_heads, head_dim].T
                        q_ij = q_spikes[b, i, :valid_steps]  # [valid_steps, num_heads, head_dim]
                        k_ij = k_spikes[b, j, :valid_steps]  # [valid_steps, num_heads, head_dim]
                        
                        # Sum over time dimension
                        for h in range(self.num_heads):
                            attention_scores[b, h, i, j] = torch.sum(
                                torch.sum(q_ij[:, h] * k_ij[:, h], dim=-1)
                            )
        
        # Scale and apply softmax
        attention_scores = attention_scores * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values (simplified - use mean over time)
        v_mean = torch.mean(v_spikes, dim=2)  # [batch_size, seq_len, num_heads, head_dim]
        v_mean = v_mean.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention output
        out = torch.matmul(attention_weights, v_mean)  # [batch_size, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        out = out.view(batch_size, seq_len, embedding_dim)
        
        return self.out_proj(out), attention_weights
    
    def compute_regularization_loss(self, T_i: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss to encourage shorter windows"""
        T_mean = T_i.float().mean()
        return self.lambda_reg * T_mean
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with adaptive spiking windows
        x: [batch_size, seq_len, embedding_dim]
        Returns: (output, metrics)
        """
        batch_size, seq_len, embedding_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x) 
        v = self.v_proj(x)
        
        # Get adaptive time windows
        T_i = self.get_adaptive_windows(x)  # [batch_size, seq_len]
        
        # Generate adaptive spikes
        q_spikes = self.generate_adaptive_spikes(q, T_i, self.lif_q)
        k_spikes = self.generate_adaptive_spikes(k, T_i, self.lif_k)
        v_spikes = self.generate_adaptive_spikes(v, T_i, self.lif_v)
        
        # Compute attention with masked accumulation
        output, attention_weights = self.masked_attention_accumulation(q_spikes, k_spikes, v_spikes, T_i)
        
        # Compute regularization loss
        reg_loss = self.compute_regularization_loss(T_i)
        
        # Store statistics for analysis
        if self.training:
            self.T_history.append(T_i.cpu().numpy())
            self.step_count += 1
        
        # Metrics for logging
        metrics = {
            'T_mean': T_i.float().mean().item(),
            'T_std': T_i.float().std().item(),
            'T_min': T_i.min().item(),
            'T_max': T_i.max().item(),
            'reg_loss': reg_loss.item(),
            'attention_weights': attention_weights.detach(),
            'adaptive_windows': T_i.detach()
        }
        
        return output, metrics
    
    def plot_analysis(self, save_path: Optional[str] = None):
        """Plot analysis of adaptive windows"""
        if not self.T_history:
            print("No history to plot. Run some forward passes first.")
            return
            
        # Combine all history
        all_T = np.concatenate(self.T_history, axis=0)  # [total_samples, seq_len]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. T_i vs token position
        T_mean_per_pos = np.mean(all_T, axis=0)
        T_std_per_pos = np.std(all_T, axis=0)
        positions = np.arange(len(T_mean_per_pos))
        
        ax1.plot(positions, T_mean_per_pos, 'b-', linewidth=2, label='Mean T_i')
        ax1.fill_between(positions, T_mean_per_pos - T_std_per_pos, 
                        T_mean_per_pos + T_std_per_pos, alpha=0.3)
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Adaptive Window Size')
        ax1.set_title('T_i vs Token Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Histogram of T_i values
        ax2.hist(all_T.flatten(), bins=range(1, self.T_max + 2), alpha=0.7, 
                density=True, edgecolor='black')
        ax2.set_xlabel('Window Size (T_i)')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of Adaptive Windows')
        ax2.grid(True, alpha=0.3)
        
        # 3. Evolution over training steps
        if len(self.T_history) > 1:
            step_means = [np.mean(T_batch) for T_batch in self.T_history]
            ax3.plot(step_means, 'g-', linewidth=2)
            ax3.set_xlabel('Training Step')
            ax3.set_ylabel('Average T_i')
            ax3.set_title('Average Window Size Over Training')
            ax3.grid(True, alpha=0.3)
        
        # 4. Heatmap of T_i across sequence
        if all_T.shape[0] > 1:
            im = ax4.imshow(all_T[:min(50, all_T.shape[0])], aspect='auto', 
                           cmap='viridis', vmin=1, vmax=self.T_max)
            ax4.set_xlabel('Token Position')  
            ax4.set_ylabel('Sample')
            ax4.set_title('Adaptive Windows Heatmap')
            plt.colorbar(im, ax=ax4, label='Window Size')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print(f"\n📊 Adaptive Spiking Windows Analysis:")
        print(f"   Overall Mean T_i: {np.mean(all_T):.2f} ± {np.std(all_T):.2f}")
        print(f"   Range: [{np.min(all_T)}, {np.max(all_T)}]")
        print(f"   Efficiency: {np.mean(all_T)/self.T_max*100:.1f}% of max window")
        print(f"   Total samples analyzed: {all_T.shape[0] * all_T.shape[1]}")


# Example usage and testing
if __name__ == "__main__":
    # Test the adaptive spiking attention
    batch_size, seq_len, embedding_dim = 2, 10, 512
    num_heads = 8
    T_max = 15
    
    # Create model
    model = AdaptiveSpikingAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads, 
        T_max=T_max,
        lambda_reg=1e-3
    )
    
    # Test input
    x = torch.randn(batch_size, seq_len, embedding_dim)
    
    print("🧠 Testing Adaptive Spiking Windows...")
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    model.train()
    output, metrics = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Metrics: {metrics}")
    
    # Run multiple steps to collect data
    print("\n🔄 Running multiple steps for analysis...")
    for step in range(20):
        x_batch = torch.randn(batch_size, seq_len, embedding_dim)
        output, metrics = model(x_batch)
        
        if step % 5 == 0:
            print(f"Step {step}: Mean T_i = {metrics['T_mean']:.2f}, "
                  f"Reg Loss = {metrics['reg_loss']:.4f}")
    
    # Plot analysis
    print("\n📈 Generating analysis plots...")
    model.plot_analysis()