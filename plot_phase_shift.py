import torch
import matplotlib.pyplot as plt
import os
import sys

# Add the project root to sys.path
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from novel_phases.phase3.snn_dt import SNNDT

def plot_phase_shift_parameters(checkpoint_path, output_path):
    """
    Loads a trained SNNDT model and plots the learned phase-shift parameters (ωk, ϕk).

    Args:
        checkpoint_path (str): The path to the model checkpoint file.
        output_path (str): The path to save the output plot.
    """
    # Define the model architecture. The parameters should match the ones used for training.
    # We can load the state_dict and infer the parameters, but for now, we'll hardcode them.
    # These are the default values from main_training.py
    embed_dim = 128
    num_heads = 4
    window_length = 10
    num_layers = 2
    
    # Instantiate the model with use_pos_encoder=True to ensure pos_encoder is created
    model = SNNDT(
        embed_dim=embed_dim,
        num_heads=num_heads,
        window_length=window_length,
        num_layers=num_layers,
        use_pos_encoder=True,
        use_router=True  # or False, it doesn't matter for this plot
    )

    # Load the state dictionary
    state_dict = torch.load(checkpoint_path)

    # Filter out unexpected keys
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    
    # Load the filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)

    # Extract the phase-shift parameters
    if hasattr(model, 'pos_encoder') and model.pos_encoder is not None:
        frequencies = model.pos_encoder.freq.detach().numpy()
        phases = model.pos_encoder.phase.detach().numpy()
    else:
        print("The model does not have a 'pos_encoder' or it is None.")
        return

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(frequencies, phases, alpha=0.7, edgecolors='b')
    plt.title('Learned Phase-Shift Parameters')
    plt.xlabel('Frequency (ωk)')
    plt.ylabel('Phase (ϕk)')
    plt.grid(True)
    
    # Save the plot
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    checkpoint_to_plot = 'novel_phases/phase3/checkpoints_phase3/full/model_final.pt'
    output_plot_path = 'novel_phases/phase3/phase_shift_parameters.png'
    
    if not os.path.exists(checkpoint_to_plot):
        print(f"Checkpoint file not found at {checkpoint_to_plot}")
        print("Please provide a valid path to a model checkpoint.")
    else:
        plot_phase_shift_parameters(checkpoint_to_plot, output_plot_path)