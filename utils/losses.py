"""
Advanced Loss Functions for EfficientMicroNet Training.

Includes:
1. Label Smoothing Cross Entropy
2. Focal Loss for imbalanced datasets
3. Knowledge Distillation Loss
4. Combined losses for better training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing.
    
    Prevents the model from becoming overconfident by softening
    the target distribution. This improves generalization.
    """
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        with torch.no_grad():
            smooth_targets = torch.zeros_like(pred)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -smooth_targets * log_preds
        loss = loss.sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Down-weights well-classified examples and focuses on hard,
    misclassified examples. Useful for imbalanced datasets.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for training smaller models.
    
    Combines hard label loss with soft label loss from a teacher model.
    The soft labels contain rich information about class relationships.
    """
    def __init__(self, temperature=4.0, alpha=0.9):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        distillation_loss = F.kl_div(
            soft_student, 
            soft_targets, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        hard_loss = self.ce_loss(student_logits, targets)
        
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        return total_loss


class MixUpLoss(nn.Module):
    """
    Loss function for MixUp and CutMix augmentation.
    
    Handles the mixed labels from MixUp/CutMix by computing
    weighted combination of losses for both original labels.
    """
    def __init__(self, base_criterion=None):
        super().__init__()
        self.base_criterion = base_criterion or nn.CrossEntropyLoss()

    def forward(self, pred, targets_a, targets_b, lam):
        loss_a = self.base_criterion(pred, targets_a)
        loss_b = self.base_criterion(pred, targets_b)
        return lam * loss_a + (1 - lam) * loss_b


class CombinedLoss(nn.Module):
    """
    Combined loss function with multiple components.
    
    Combines label smoothing, focal loss weighting, and optional
    knowledge distillation for comprehensive training.
    """
    def __init__(self, smoothing=0.1, focal_gamma=0.0, use_focal=False):
        super().__init__()
        self.smoothing = smoothing
        self.use_focal = use_focal
        
        if use_focal:
            self.criterion = FocalLoss(gamma=focal_gamma)
        else:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)

    def forward(self, pred, target, targets_b=None, lam=1.0):
        if targets_b is not None and lam < 1.0:
            loss_a = self.criterion(pred, target)
            loss_b = self.criterion(pred, targets_b)
            return lam * loss_a + (1 - lam) * loss_b
        return self.criterion(pred, target)
