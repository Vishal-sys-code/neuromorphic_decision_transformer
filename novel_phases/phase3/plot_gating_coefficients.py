import torch
import matplotlib.pyplot as plt
import numpy as np

from dendritic_routing import DendriticRouter

def plot_gating_coefficients(gates, save_path=None):
    """
    Plots the gating coefficients as a heatmap.

    Args:
        gates (torch.Tensor): A tensor of shape [B, L, H] representing the gating coefficients.
        save_path (str, optional): The path to save the heatmap image. Defaults to None.
    """
    # Assuming we are plotting the first sample in the batch
    gates_to_plot = gates[0].detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(gates_to_plot.T, cmap='viridis', aspect='auto')
    plt.colorbar(label='Gating Coefficient ($\\alpha_i^{(h)}$)')
    plt.xlabel('Token Index (i)')
    plt.ylabel('Head Index (h)')
    plt.title('Heatmap of Gating Coefficients')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

if __name__ == '__main__':
    # Parameters
    B, L, H, d = 1, 50, 8, 128 # Batch size, sequence length, number of heads, feature dimension

    # Model
    model = DendriticRouter(num_heads=H)

    # Input
    y_heads_input = torch.randn(B, L, H, d)

    # Get gates
    summary = y_heads_input.sum(dim=-1)
    gates = model.routing_mlp(summary.view(-1, H)).view(B, L, H)

    # Plot
    plot_gating_coefficients(gates, save_path='gating_coefficients_heatmap.png')
