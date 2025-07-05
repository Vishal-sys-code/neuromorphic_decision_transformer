import torch
# Assuming your models are in SpikingMindRL.src.models
# Adjust the import path if your project structure is different
# For example, if SpikingMindRL is the top-level package recognized by Python:
# from SpikingMindRL.src.models.snn_dt import SNNDT
# If 'src' is the directory you run from and is in PYTHONPATH:
# from models.snn_dt import SNNDT
from src.models.snn_dt import SNNDT
# If SpikingMindRL is in PYTHONPATH and src is a package:
# from src.models.snn_dt import SNNDT

def main():
    print("Starting main training script...")

    # Configuration (replace with your actual configuration loading)
    batch_size = 16
    seq_length = 50
    embed_dim = 128
    num_heads = 4
    window_length = 10 # T
    num_layers = 2    # Number of SNN layers in SNNDT
    learning_rate = 1e-4
    num_epochs = 100

    # Initialize the model
    # Ensure these parameters match those expected by your SNNDT's __init__
    model = SNNDT(
        embed_dim=embed_dim,
        num_heads=num_heads,
        window_length=window_length,
        num_layers=num_layers
    )

    # Dummy data (replace with your actual data loading and preprocessing)
    # Input embeddings: [B, L, d]
    dummy_input_embeddings = torch.randn(batch_size, seq_length, embed_dim)
    # Target data (shape depends on your task, e.g., next token prediction, classification)
    # For example, if predicting a sequence of the same dimension:
    dummy_targets = torch.randn(batch_size, seq_length, embed_dim)
    # Or for classification, targets might be class indices:
    # dummy_targets_classification = torch.randint(0, num_classes, (batch_size,))

    # Loss function (choose based on your task)
    # For regression-like tasks or predicting continuous embeddings:
    criterion = torch.nn.MSELoss()
    # For classification:
    # criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Model initialized: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Training loop
    for epoch in range(num_epochs):
        model.train() # Set model to training mode

        # Forward pass
        outputs = model(dummy_input_embeddings) # [B, L, d]

        # Calculate loss
        # Ensure shapes of outputs and targets match what criterion expects
        loss = criterion(outputs, dummy_targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Validation step (highly recommended)
        if (epoch + 1) % 10 == 0: # Example: validate every 10 epochs
            model.eval() # Set model to evaluation mode
            with torch.no_grad():
                # Use a validation dataset here
                val_outputs = model(dummy_input_embeddings) # Replace with validation data
                val_loss = criterion(val_outputs, dummy_targets) # Replace with validation targets
                print(f"Validation Loss after Epoch {epoch+1}: {val_loss.item():.4f}")
            
            # Inspect learned parameters (as suggested in the prompt)
            if hasattr(model, 'pos_encoder'):
                print(f"  Positional Encoder Freqs: {model.pos_encoder.freq.data.numpy().round(3)}")
                print(f"  Positional Encoder Phases: {model.pos_encoder.phase.data.numpy().round(3)}")
            if hasattr(model, 'router') and hasattr(model.router, 'routing_mlp'):
                 # Accessing the first linear layer of the sequential MLP
                if len(model.router.routing_mlp) > 0 and isinstance(model.router.routing_mlp[0], torch.nn.Linear):
                    print(f"  Router MLP Layer 1 Weights (sample): {model.router.routing_mlp[0].weight.data[0,:5].numpy().round(3)}")


    print("Training finished.")

    # TODO: Add code for saving the model, further evaluation, etc.

if __name__ == "__main__":
    # This basic structure assumes you might run this script directly.
    # For pytest, you'd typically import functions or classes to test,
    # not run the main training loop.
    
    # To make this runnable, ensure SpikingMindRL/src/models/ is accessible.
    # If you run from SpikingMindRL/src/, the import `from models.snn_dt import SNNDT` should work.
    # If you run from SpikingMindRL/, you might need to adjust python path or use:
    # `python -m src.main_training` and ensure src has an __init__.py
    
    # For now, let's create __init__.py files to make them importable as packages.
    main()