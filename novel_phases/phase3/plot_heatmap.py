import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from snn_dt import SNNDT
import os

def plot_routing_heatmap(gate_data, save_path="routing_heatmap.png"):
    """
    Plots a heatmap of routing coefficients (gates).

    Args:
        gate_data (torch.Tensor): A tensor of shape [num_layers, batch_size, seq_len, num_heads]
                                  or [batch_size, seq_len, num_heads] for a single layer.
        save_path (str): Path to save the generated heatmap image.
    """
    if gate_data.dim() == 4:
        # Taking the first layer and first batch item for visualization
        gate_data = gate_data[0, 0, :, :]
    elif gate_data.dim() == 3:
        # Taking the first batch item
        gate_data = gate_data[0, :, :]
    else:
        raise ValueError("gate_data must have 3 or 4 dimensions")

    gate_data_np = gate_data.detach().cpu().numpy()

    plt.figure(figsize=(12, 8))
    sns.heatmap(gate_data_np.T, cmap="viridis", cbar_kws={'label': 'Routing Coefficient Value'})
    plt.xlabel("Token Position (t)")
    plt.ylabel("Attention Head (h)")
    plt.title("Heatmap of Routing Coefficients α(h)(t) for a Representative Sequence")
    plt.savefig(save_path)
    plt.close()
    print(f"Heatmap saved to {save_path}")

def main():
    # Model configuration (should match the model you want to visualize)
    embed_dim = 128
    num_heads = 8
    window_length = 10
    num_layers = 4
    seq_length = 50
    batch_size = 1 # Using a single item for easy visualization

    # Initialize the model with router enabled
    model = SNNDT(
        embed_dim=embed_dim,
        num_heads=num_heads,
        window_length=window_length,
        num_layers=num_layers,
        use_router=True,
        use_pos_encoder=True
    )
    model.eval()

    # Create dummy input data
    dummy_input = torch.randn(batch_size, seq_length, embed_dim)

    # Perform a forward pass and get the gates
    with torch.no_grad():
        _, gates = model(dummy_input, return_gates=True)

    # Define the output path
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "routing_coefficients_heatmap.png")

    # Plot the heatmap
    plot_routing_heatmap(gates, save_path=save_path)

if __name__ == "__main__":
    main()