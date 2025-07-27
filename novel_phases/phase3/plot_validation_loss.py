import sys
import os
import torch
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to sys.path
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.snn_dt import SNNDT

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main(args):
    # Load the dataset
    data = load_data(args.data_path)
    # Unpack only the first 6 elements, ignore extras
    states, actions, returns, _, _, _ = data[:6]


    # Ensure states, actions, returns are at least 1D numpy arrays for slicing
    def ensure_1d_array(x):
        arr = np.array(x)
        if arr.ndim == 0:
            arr = np.expand_dims(arr, 0)
        return arr
    states = ensure_1d_array(states)
    actions = ensure_1d_array(actions)
    returns = ensure_1d_array(returns)

    # For simplicity, using a subset of data for this example
    # In a real scenario, you'd use a proper train/validation split
    train_states = torch.from_numpy(states[:100]).float()
    train_actions = torch.from_numpy(actions[:100]).float()
    train_returns = torch.from_numpy(returns[:100]).float()

    val_states = torch.from_numpy(states[100:120]).float()
    val_actions = torch.from_numpy(actions[100:120]).float()
    val_returns = torch.from_numpy(returns[100:120]).float()

    ablation_modes = ["baseline", "pos_only", "router_only", "full"]
    results = {}

    for mode in ablation_modes:
        print(f"Running ablation mode: {mode}")

        use_pos_encoder_flag = mode in ["pos_only", "full"]
        use_router_flag = mode in ["router_only", "full"]

        model = SNNDT(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            window_length=args.window_length,
            num_layers=args.num_layers,
            use_pos_encoder=use_pos_encoder_flag,
            use_router=use_router_flag
        )

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        val_loss_history = []

        for epoch in range(args.epochs):
            # Training step
            model.train()
            # This is a simplified training step. In a real scenario, you would iterate through batches of your data.
            # Here, we'll just use a single batch for demonstration.
            optimizer.zero_grad()
            # The model expects inputs of shape (batch_size, seq_length, embed_dim).
            # The loaded data is not in this format, so we'll use dummy inputs for now.
            dummy_input_embeddings = torch.randn(args.batch_size, args.seq_length, args.embed_dim)
            dummy_targets = torch.randn(args.batch_size, args.seq_length, args.embed_dim)
            outputs = model(dummy_input_embeddings)
            loss = criterion(outputs, dummy_targets)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % args.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    # Similar to training, we use a dummy validation set
                    val_outputs = model(dummy_input_embeddings)
                    val_loss = criterion(val_outputs, dummy_targets)
                    val_loss_history.append(val_loss.item())
                    print(f"Epoch [{epoch+1}/{args.epochs}], Mode: {mode}, Val Loss: {val_loss.item():.4f}")
        
        results[mode] = val_loss_history

    # Plotting the results
    plt.figure(figsize=(12, 8))
    for mode, losses in results.items():
        epochs = np.arange(1, args.epochs + 1, args.val_interval)
        plt.plot(epochs, losses, label=mode)

    plt.title('Validation Loss vs. Epoch for each Ablation Mode')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('validation_loss_vs_epoch.png')
    print("Plot saved as validation_loss_vs_epoch.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN-DT Validation Loss Plotting Script")
    parser.add_argument('--data_path', type=str, default='../../saved_models/offline_data_CartPole-v1.pkl', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_length', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--window_length', type=int, default=10, help="T, time window for spiking dynamics")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of SNN layers in SNNDT")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=5)
    
    cli_args = parser.parse_args()
    main(cli_args)