import sys
import os

# Add the project root to sys.path
# This assumes main_training.py is in SpikingMindRL/novel_phases/phase3/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now the import should work
from src.models.snn_dt import SNNDT
import torch
import argparse

def main(args):
    print(f"Starting main training script with args: {args}")

    # Configuration (replace with your actual configuration loading or use args)
    batch_size = args.batch_size
    seq_length = args.seq_length
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    window_length = args.window_length # T
    num_layers = args.num_layers    # Number of SNN layers in SNNDT
    learning_rate = args.lr
    num_epochs = args.epochs
    
    use_pos_encoder_flag = True
    use_router_flag = True

    if args.ablation_mode == "baseline":
        use_pos_encoder_flag = False
        use_router_flag = False
        print("Running in ABLATION MODE: BASELINE (Rate coding only)")
    elif args.ablation_mode == "pos_only":
        use_pos_encoder_flag = True
        use_router_flag = False
        print("Running in ABLATION MODE: +POSITIONAL SPIKES (Positional Encoder ON, Router OFF)")
    elif args.ablation_mode == "router_only":
        use_pos_encoder_flag = False
        use_router_flag = True
        print("Running in ABLATION MODE: +ROUTING ONLY (Positional Encoder OFF, Router ON)")
    elif args.ablation_mode == "full":
        use_pos_encoder_flag = True
        use_router_flag = True
        print("Running in ABLATION MODE: FULL (Positional Encoder ON, Router ON)")
    else:
        print("Warning: Unknown ablation mode. Defaulting to full configuration.")


    # Initialize the model
    model = SNNDT(
        embed_dim=embed_dim,
        num_heads=num_heads,
        window_length=window_length,
        num_layers=num_layers,
        use_pos_encoder=use_pos_encoder_flag,
        use_router=use_router_flag
    )

    # Dummy data (replace with your actual data loading and preprocessing)
    dummy_input_embeddings = torch.randn(batch_size, seq_length, embed_dim)
    dummy_targets = torch.randn(batch_size, seq_length, embed_dim)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Model initialized for mode: {args.ablation_mode}")
    print(f"  Positional Encoder: {'Enabled' if use_pos_encoder_flag else 'Disabled'}")
    print(f"  Router: {'Enabled' if use_router_flag else 'Disabled'}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    log_dir = os.path.join(args.log_dir, args.ablation_mode)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.ablation_mode)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Logging to: {log_dir}")
    print(f"Saving checkpoints to: {checkpoint_dir}")

    # Training loop
    for epoch in range(num_epochs):
        model.train() 
        outputs = model(dummy_input_embeddings)
        loss = criterion(outputs, dummy_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Mode: {args.ablation_mode}, Loss: {loss.item():.4f}")

        if (epoch + 1) % args.val_interval == 0:
            model.eval() 
            with torch.no_grad():
                val_outputs = model(dummy_input_embeddings) 
                val_loss = criterion(val_outputs, dummy_targets)
                print(f"Validation Loss after Epoch {epoch+1} (Mode: {args.ablation_mode}): {val_loss.item():.4f}")
            
            if use_pos_encoder_flag and hasattr(model, 'pos_encoder') and model.pos_encoder is not None:
                print(f"  Positional Encoder Freqs: {model.pos_encoder.freq.data.numpy().round(3)}")
                print(f"  Positional Encoder Phases: {model.pos_encoder.phase.data.numpy().round(3)}")
            if use_router_flag and hasattr(model, 'router') and model.router is not None and hasattr(model.router, 'routing_mlp'):
                if len(model.router.routing_mlp) > 0 and isinstance(model.router.routing_mlp[0], torch.nn.Linear):
                    print(f"  Router MLP Layer 1 Weights (sample): {model.router.routing_mlp[0].weight.data[0,:5].numpy().round(3)}")
        
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


    print(f"Training finished for mode: {args.ablation_mode}.")
    final_model_path = os.path.join(checkpoint_dir, "model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN-DT Training Script with Ablations")
    parser.add_argument('--ablation_mode', type=str, default="full",
                        choices=["baseline", "pos_only", "router_only", "full"],
                        help='Ablation mode to run.')
    # Basic training params (can be overridden by a config file later)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_length', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--window_length', type=int, default=10, help="T, time window for spiking dynamics")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of SNN layers in SNNDT")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100) # Reduced for quick test, original was 100
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default="logs_phase3")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints_phase3")
    parser.add_argument('--val_interval', type=int, default=5, help="Epoch interval for validation") # original 10
    parser.add_argument('--save_interval', type=int, default=10, help="Epoch interval for saving checkpoints") # new

    cli_args = parser.parse_args()
    
    # Ensure src.models.snn_dt is accessible
    main(cli_args)