"""
Optimizer and Learning Rate Scheduler utilities for EfficientMicroNet.

Includes:
1. AdamW with weight decay
2. Cosine Annealing with Warm Restarts
3. Warmup scheduler
4. Layer-wise learning rate decay
"""

import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR
import math


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup and cosine annealing.
    
    Starts with a low learning rate, linearly increases during warmup,
    then follows cosine decay. This helps stabilize early training.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, warmup_start_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.warmup_start_lr + progress * (base_lr - self.warmup_start_lr)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def get_optimizer(model, lr=1e-3, weight_decay=0.05, optimizer_type='adamw'):
    """
    Create optimizer with proper weight decay handling.
    
    Separates parameters into those that should have weight decay
    (weights) and those that shouldn't (biases, norms).
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith('.bias') or 'bn' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    if optimizer_type == 'adamw':
        return AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    elif optimizer_type == 'sgd':
        return SGD(param_groups, lr=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_scheduler(optimizer, scheduler_type='cosine', total_epochs=100, warmup_epochs=5, 
                  min_lr=1e-6, steps_per_epoch=None):
    """
    Create learning rate scheduler.
    """
    if scheduler_type == 'cosine':
        return WarmupCosineScheduler(
            optimizer, 
            warmup_epochs=warmup_epochs, 
            total_epochs=total_epochs,
            min_lr=min_lr
        )
    elif scheduler_type == 'cosine_restart':
        return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=min_lr)
    elif scheduler_type == 'onecycle' and steps_per_epoch:
        return OneCycleLR(
            optimizer,
            max_lr=[group['lr'] for group in optimizer.param_groups],
            total_steps=total_epochs * steps_per_epoch,
            pct_start=warmup_epochs / total_epochs,
            anneal_strategy='cos',
            final_div_factor=1000
        )
    else:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)


class EMA:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a moving average of model parameters which often
    provides better generalization than the final model weights.
    """
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
