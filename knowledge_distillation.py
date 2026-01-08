"""
Knowledge Distillation Training for EfficientMicroNet.

Train a smaller student model using knowledge from a larger teacher model.
This technique can significantly improve the accuracy of lightweight models.
"""

import os
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, models

from models import emn_tiny, emn_small, emn_base, EfficientMicroNet
from utils import (
    get_train_transforms, get_val_transforms,
    KnowledgeDistillationLoss,
    get_optimizer, get_scheduler, EMA
)


class DistillationTrainer:
    """
    Knowledge Distillation Trainer.
    
    Trains a student model using soft labels from a teacher model
    combined with hard labels from the dataset.
    """
    
    def __init__(self, student, teacher, device, temperature=4.0, alpha=0.9):
        self.student = student
        self.teacher = teacher
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.criterion = KnowledgeDistillationLoss(temperature=temperature, alpha=alpha)
    
    def train_step(self, images, targets, optimizer, scaler, use_amp=True):
        """Single training step with distillation."""
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        with torch.no_grad():
            teacher_logits = self.teacher(images)
        
        with autocast(enabled=use_amp):
            student_logits = self.student(images)
            loss = self.criterion(student_logits, teacher_logits, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        return loss.item(), student_logits
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validate student model."""
        self.student.eval()
        
        correct = 0
        total = 0
        total_loss = 0
        
        for images, targets in val_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            student_logits = self.student(images)
            teacher_logits = self.teacher(images)
            
            loss = self.criterion(student_logits, teacher_logits, targets)
            total_loss += loss.item() * images.size(0)
            
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        self.student.train()
        
        return total_loss / total, 100.0 * correct / total


def get_teacher_model(teacher_name, num_classes, pretrained=True):
    """Load a pretrained teacher model."""
    if teacher_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif teacher_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif teacher_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif teacher_name == 'efficientnet_b3':
        model = models.efficientnet_b3(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown teacher model: {teacher_name}")
    
    return model


def main(args):
    """Main distillation training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_transform = get_train_transforms(img_size=args.img_size)
    val_transform = get_val_transforms(img_size=args.img_size)
    
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, transform=train_transform, download=True)
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=val_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir, train=True, transform=train_transform, download=True)
        val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=val_transform, download=True)
        num_classes = 100
    else:
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=val_transform)
        num_classes = len(train_dataset.classes)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    print(f'Loading teacher model: {args.teacher}')
    teacher = get_teacher_model(args.teacher, num_classes, pretrained=True)
    
    if args.teacher_checkpoint:
        teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device))
    
    teacher = teacher.to(device)
    
    if args.student == 'emn_tiny':
        student = emn_tiny(num_classes=num_classes)
    elif args.student == 'emn_small':
        student = emn_small(num_classes=num_classes)
    else:
        student = emn_base(num_classes=num_classes)
    
    student = student.to(device)
    
    print(f'\nTeacher: {args.teacher}')
    print(f'Teacher params: {sum(p.numel() for p in teacher.parameters()):,}')
    print(f'Student: {args.student}')
    print(f'Student params: {sum(p.numel() for p in student.parameters()):,}')
    
    trainer = DistillationTrainer(
        student, teacher, device,
        temperature=args.temperature,
        alpha=args.alpha
    )
    
    optimizer = get_optimizer(student, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, total_epochs=args.epochs, warmup_epochs=args.warmup_epochs)
    scaler = GradScaler(enabled=args.amp)
    
    ema = EMA(student, decay=0.9999) if args.use_ema else None
    
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    print(f'\nStarting distillation training for {args.epochs} epochs...\n')
    
    for epoch in range(args.epochs):
        student.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            loss, _ = trainer.train_step(images, targets, optimizer, scaler, args.amp)
            total_loss += loss
            num_batches += 1
            
            if ema is not None:
                ema.update()
            
            if batch_idx % args.print_freq == 0:
                print(f'Epoch [{epoch}][{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss:.4f} LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        scheduler.step(epoch)
        
        if ema is not None:
            ema.apply_shadow()
        
        val_loss, val_acc = trainer.validate(val_loader)
        
        if ema is not None:
            ema.restore()
        
        avg_train_loss = total_loss / num_batches
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        torch.save({
            'epoch': epoch + 1,
            'state_dict': student.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, output_dir / 'checkpoint.pth')
        
        if is_best:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': student.state_dict(),
                'best_acc': best_acc,
            }, output_dir / 'model_best.pth')
        
        print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Best: {best_acc:.2f}%\n')
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'\nDistillation completed! Best accuracy: {best_acc:.2f}%')
    print(f'Model saved to: {output_dir}')


def parse_args():
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training')
    
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--output-dir', type=str, default='./outputs_distill')
    
    parser.add_argument('--teacher', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'efficientnet_b0', 'efficientnet_b3'])
    parser.add_argument('--teacher-checkpoint', type=str, default='')
    parser.add_argument('--student', type=str, default='emn_tiny',
                        choices=['emn_tiny', 'emn_small', 'emn_base'])
    
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--alpha', type=float, default=0.9)
    
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    
    parser.add_argument('--use-ema', action='store_true', default=True)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--print-freq', type=int, default=50)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
