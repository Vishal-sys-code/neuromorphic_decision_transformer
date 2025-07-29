import sys
import os
import time
import torch
import argparse
import pandas as pd

# Add the project root to sys.path
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from novel_phases.phase3.snn_dt import SNNDT

def run_benchmark(args):
    # Configuration
    batch_size = 1  # Benchmark with a single inference
    seq_length = 50
    embed_dim = 128
    num_heads = 4
    window_length = 10
    num_layers = 2

    use_pos_encoder_flag = True
    use_router_flag = True

    if args.ablation_mode == "baseline":
        use_pos_encoder_flag = False
        use_router_flag = False
    elif args.ablation_mode == "pos_only":
        use_pos_encoder_flag = True
        use_router_flag = False
    elif args.ablation_mode == "router_only":
        use_pos_encoder_flag = False
        use_router_flag = True
    elif args.ablation_mode == "full":
        use_pos_encoder_flag = True
        use_router_flag = True

    # Initialize the model
    model = SNNDT(
        embed_dim=embed_dim,
        num_heads=num_heads,
        window_length=window_length,
        num_layers=num_layers,
        use_pos_encoder=use_pos_encoder_flag,
        use_router=use_router_flag
    )

    # Load the pre-trained model
    checkpoint_path = os.path.join(args.checkpoint_dir, args.ablation_mode, "model_final.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Dummy data
    dummy_input_embeddings = torch.randn(batch_size, seq_length, embed_dim)

    # Warm-up run
    with torch.no_grad():
        _ = model(dummy_input_embeddings, return_spikes=False)

    # Measure latency and spikes
    with torch.no_grad():
        start_time = time.time()
        _, spikes = model(dummy_input_embeddings, return_spikes=True)
        end_time = time.time()

    latency_ms = (end_time - start_time) * 1000
    total_spikes = spikes.sum().item()

    # Print results in CSV format
    print(f"{args.ablation_mode},{total_spikes},{latency_ms:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN-DT Benchmark Script")
    parser.add_argument('--ablation_mode', type=str, required=True,
                        choices=["baseline", "pos_only", "router_only", "full"],
                        help='Ablation mode to benchmark.')
    parser.add_argument('--checkpoint_dir', type=str, default="novel_phases/phase3/checkpoints_phase3",
                        help='Directory where checkpoints are stored.')
    
    cli_args = parser.parse_args()
    
    run_benchmark(cli_args)