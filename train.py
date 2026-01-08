"""
Training Script for EfficientMicroNet.

Features:
- Mixed precision training (AMP)
- Gradient accumulation
- EMA (Exponential Moving Average)
- CutMix/MixUp augmentation
- Knowledge distillation support
- Comprehensive logging
- Checkpoint saving/resuming
"""

import os
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets

from models import emn_tiny, emn_small, emn_base, EfficientMicroNet
from utils import (
    get_train_transforms, get_val_transforms,
    CutMixMixUp, ProgressiveResizing,
    CombinedLoss, KnowledgeDistillationLoss,
    get_optimizer, get_scheduler, EMA
)


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, 
                    scaler, device, epoch, args, ema=None, mixup_fn=None):
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    
    end = time.time()
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        targets_a, targets_b, lam = targets, targets, 1.0
        if mixup_fn is not None:
            images, targets_a, targets_b, lam = mixup_fn(images, targets)
        
        with autocast(enabled=args.amp):
            outputs = model(images)
            loss = criterion(outputs, targets_a, targets_b, lam)
        
        loss = loss / args.grad_accum_steps
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % args.grad_accum_steps == 0:
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if ema is not None:
                ema.update()
        
        acc1, acc5 = accuracy(outputs, targets_a, topk=(1, min(5, args.num_classes)))
        
        losses.update(loss.item() * args.grad_accum_steps, images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] '
                  f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'Loss: {losses.val:.4f} ({losses.avg:.4f}) '
                  f'Acc@1: {top1.val:.2f} ({top1.avg:.2f}) '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return losses.avg, top1.avg, top5.avg


@torch.no_grad()
def validate(model, val_loader, criterion, device, args):
    """Validate the model."""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with autocast(enabled=args.amp):
            outputs = model(images)
            loss = criterion(outputs, targets, targets, 1.0)
        
        acc1, acc5 = accuracy(outputs, targets, topk=(1, min(5, args.num_classes)))
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))
    
    print(f'Validation: Loss: {losses.avg:.4f} Acc@1: {top1.avg:.2f} Acc@5: {top5.avg:.2f}')
    
    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth'):
    """Save checkpoint."""
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(output_dir, 'model_best.pth')
        torch.save(state, best_path)


def main(args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    train_transform = get_train_transforms(
        img_size=args.img_size,
        auto_augment=args.auto_augment,
        randaugment_n=args.randaugment_n,
        randaugment_m=args.randaugment_m
    )
    val_transform = get_val_transforms(img_size=args.img_size)
    
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, transform=train_transform, download=True)
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=val_transform, download=True)
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir, train=True, transform=train_transform, download=True)
        val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=val_transform, download=True)
        args.num_classes = 100
    elif args.dataset == 'imagenet':
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=val_transform)
        args.num_classes = 1000
    else:
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=val_transform)
        args.num_classes = len(train_dataset.classes)
    
    print(f'Dataset: {args.dataset}, Classes: {args.num_classes}')
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    if args.model == 'emn_tiny':
        model = emn_tiny(num_classes=args.num_classes, dropout=args.dropout)
    elif args.model == 'emn_small':
        model = emn_small(num_classes=args.num_classes, dropout=args.dropout)
    elif args.model == 'emn_base':
        model = emn_base(num_classes=args.num_classes, dropout=args.dropout)
    else:
        model = EfficientMicroNet(
            num_classes=args.num_classes,
            variant=args.variant,
            width_mult=args.width_mult,
            dropout=args.dropout
        )
    
    model = model.to(device)
    
    print(f'Model: {args.model}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    criterion = CombinedLoss(smoothing=args.label_smoothing, use_focal=args.use_focal)
    
    optimizer = get_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay, 
        optimizer_type=args.optimizer
    )
    
    scheduler = get_scheduler(
        optimizer, scheduler_type=args.scheduler,
        total_epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr
    )
    
    scaler = GradScaler(enabled=args.amp)
    
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    
    mixup_fn = CutMixMixUp(
        cutmix_alpha=args.cutmix_alpha,
        mixup_alpha=args.mixup_alpha,
        prob=args.mixup_prob
    ) if args.use_mixup else None
    
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'Loading checkpoint: {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            print(f'Resumed from epoch {start_epoch}, best_acc: {best_acc:.2f}')
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    print(f'\nStarting training for {args.epochs} epochs...\n')
    
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, args, ema, mixup_fn
        )
        
        scheduler.step(epoch)
        
        if ema is not None:
            ema.apply_shadow()
        
        val_loss, val_acc, _ = validate(model, val_loader, criterion, device, args)
        
        if ema is not None:
            ema.restore()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
        }, is_best, output_dir)
        
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}, Best: {best_acc:.2f}\n')
    
    print(f'\nTraining completed! Best accuracy: {best_acc:.2f}%')
    print(f'Model saved to: {output_dir}')


def parse_args():
    parser = argparse.ArgumentParser(description='Train EfficientMicroNet')
    
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100', 'imagenet', 'custom'])
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    
    parser.add_argument('--model', type=str, default='emn_small',
                        choices=['emn_tiny', 'emn_small', 'emn_base', 'custom'])
    parser.add_argument('--variant', type=str, default='small', choices=['tiny', 'small', 'base'])
    parser.add_argument('--width-mult', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['cosine', 'cosine_restart', 'onecycle'])
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--use-focal', action='store_true')
    
    parser.add_argument('--auto-augment', action='store_true', default=True)
    parser.add_argument('--randaugment-n', type=int, default=2)
    parser.add_argument('--randaugment-m', type=int, default=9)
    parser.add_argument('--use-mixup', action='store_true', default=True)
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--cutmix-alpha', type=float, default=1.0)
    parser.add_argument('--mixup-prob', type=float, default=0.5)
    
    parser.add_argument('--use-ema', action='store_true', default=True)
    parser.add_argument('--ema-decay', type=float, default=0.9999)
    
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--print-freq', type=int, default=50)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
