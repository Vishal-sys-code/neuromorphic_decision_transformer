import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

class AdaptiveWindowAnalyzer:
    def __init__(self):
        self.history = []
        
    def collect_metrics(self, all_metrics: List[Dict]):
        """Collect metrics from all transformer layers"""
        batch_data = {
            'T_means': [m['T_mean'] for m in all_metrics],
            'T_stds': [m['T_std'] for m in all_metrics],
            'reg_losses': [m['reg_loss'] for m in all_metrics],
            'adaptive_windows': [m['adaptive_windows'].cpu().numpy() 
                               for m in all_metrics]
        }
        self.history.append(batch_data)
    
    def plot_layer_comparison(self, save_path=None):
        """Compare adaptive windows across transformer layers"""
        if not self.history:
            return
            
        num_layers = len(self.history[0]['T_means'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data across all batches
        layer_means = [[] for _ in range(num_layers)]
        for batch in self.history:
            for layer_idx, mean_val in enumerate(batch['T_means']):
                layer_means[layer_idx].append(mean_val)
        
        # Plot 1: Average window size per layer over time
        for layer_idx in range(num_layers):
            axes[0,0].plot(layer_means[layer_idx], 
                          label=f'Layer {layer_idx+1}', alpha=0.8)
        axes[0,0].set_title('Average Window Size Over Training')
        axes[0,0].set_xlabel('Training Step')
        axes[0,0].set_ylabel('Mean T_i')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Final window size distribution per layer
        final_means = [layer_means[i][-1] for i in range(num_layers)]
        axes[0,1].bar(range(1, num_layers+1), final_means, alpha=0.7)
        axes[0,1].set_title('Final Average Window Size by Layer')
        axes[0,1].set_xlabel('Layer')
        axes[0,1].set_ylabel('Mean T_i')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Regularization loss per layer
        layer_reg_losses = [[] for _ in range(num_layers)]
        for batch in self.history:
            for layer_idx, reg_loss in enumerate(batch['reg_losses']):
                layer_reg_losses[layer_idx].append(reg_loss)
                
        for layer_idx in range(num_layers):
            axes[1,0].plot(layer_reg_losses[layer_idx], 
                          label=f'Layer {layer_idx+1}', alpha=0.8)
        axes[1,0].set_title('Regularization Loss Over Training')
        axes[1,0].set_xlabel('Training Step')
        axes[1,0].set_ylabel('Reg Loss')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Efficiency metrics
        efficiencies = []
        for batch in self.history:
            batch_eff = [mean/20.0 for mean in batch['T_means']]  # Assuming T_max=20
            efficiencies.append(np.mean(batch_eff))
            
        axes[1,1].plot(efficiencies, 'g-', linewidth=2)
        axes[1,1].set_title('Model Efficiency Over Training')
        axes[1,1].set_xlabel('Training Step')
        axes[1,1].set_ylabel('Efficiency (avg T_i / T_max)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print(f"\n📊 Multi-Layer Adaptive Windows Analysis:")
        print(f"   Total training steps: {len(self.history)}")
        print(f"   Number of layers: {num_layers}")
        for i in range(num_layers):
            final_mean = layer_means[i][-1] if layer_means[i] else 0
            print(f"   Layer {i+1} final avg window: {final_mean:.2f}")
        print(f"   Final model efficiency: {efficiencies[-1]*100:.1f}%")