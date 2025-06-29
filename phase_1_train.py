import torch
from src.models.full_model import AdaptiveSpikingTransformer
from src.training.trainer import AdaptiveSpikingTrainer
from src.
from src.utils.analysis import AdaptiveWindowAnalyzer

def main():
    # Model configuration
    model = AdaptiveSpikingTransformer(
        vocab_size=10000,
        embedding_dim=512,
        num_heads=8,
        num_layers=6,
        T_max=20,          # 🔥 Maximum temporal window
        lambda_reg=1e-3    # 🔥 Regularization strength
    )
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = AdaptiveSpikingTrainer(model, optimizer)
    analyzer = AdaptiveWindowAnalyzer()
    
    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            
            # Training step
            train_metrics = trainer.train_step(batch)
            
            # Get model metrics for analysis
            with torch.no_grad():
                _, all_metrics = model(batch[0])
                analyzer.collect_metrics(all_metrics)
            
            # Log progress
            if trainer.step_count % 100 == 0:
                print(f"Step {trainer.step_count}: "
                      f"Loss = {train_metrics['total_loss']:.4f}, "
                      f"Avg Window = {train_metrics['avg_window_size']:.2f}")
        
        # 🔥 Generate analysis plots every epoch
        if epoch % 5 == 0:
            analyzer.plot_layer_comparison(
                save_path=f'plots/analysis_epoch_{epoch}.png'
            )

if __name__ == "__main__":
    main()