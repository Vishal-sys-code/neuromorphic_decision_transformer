import torch
import torch.nn as nn
from ..models.full_model import AdaptiveSpikingTransformer

class AdaptiveSpikingTrainer:
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.step_count = 0
        
    def train_step(self, batch):
        self.model.train()
        
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Forward pass
        logits, all_metrics = self.model(inputs)
        
        # Task loss (e.g., cross-entropy)
        task_loss = nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1)
        )
        
        # 🔥 Collect regularization losses from all layers
        reg_loss = sum([metrics['reg_loss'] for metrics in all_metrics])
        
        # Total loss
        total_loss = task_loss + reg_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # 🔥 Log adaptive window statistics
        avg_T_mean = sum([m['T_mean'] for m in all_metrics]) / len(all_metrics)
        
        self.step_count += 1
        
        return {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(), 
            'reg_loss': reg_loss.item(),
            'avg_window_size': avg_T_mean,
            'step': self.step_count
        }