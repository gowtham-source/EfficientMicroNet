"""
Evaluation Script for EfficientMicroNet.

Features:
- Comprehensive metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- Per-class accuracy analysis
- Model efficiency metrics (FLOPs, latency)
- Grad-CAM visualization for interpretability
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
from torchvision import datasets
import numpy as np

from models import emn_tiny, emn_small, emn_base, EfficientMicroNet
from utils import get_val_transforms


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model, device, num_classes):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """Run full evaluation on dataset."""
        all_preds = []
        all_targets = []
        all_probs = []
        
        for images, targets in dataloader:
            images = images.to(self.device)
            outputs = self.model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        metrics = self._compute_metrics(all_preds, all_targets, all_probs)
        return metrics
    
    def _compute_metrics(self, preds, targets, probs):
        """Compute comprehensive metrics."""
        metrics = {}
        
        metrics['accuracy'] = (preds == targets).mean() * 100
        
        metrics['top5_accuracy'] = self._top_k_accuracy(probs, targets, k=min(5, self.num_classes))
        
        per_class_acc = []
        per_class_precision = []
        per_class_recall = []
        per_class_f1 = []
        
        for c in range(self.num_classes):
            class_mask = targets == c
            if class_mask.sum() > 0:
                class_acc = (preds[class_mask] == c).mean() * 100
                per_class_acc.append(class_acc)
                
                tp = ((preds == c) & (targets == c)).sum()
                fp = ((preds == c) & (targets != c)).sum()
                fn = ((preds != c) & (targets == c)).sum()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                per_class_precision.append(precision)
                per_class_recall.append(recall)
                per_class_f1.append(f1)
        
        metrics['per_class_accuracy'] = per_class_acc
        metrics['mean_class_accuracy'] = np.mean(per_class_acc)
        metrics['macro_precision'] = np.mean(per_class_precision) * 100
        metrics['macro_recall'] = np.mean(per_class_recall) * 100
        metrics['macro_f1'] = np.mean(per_class_f1) * 100
        
        metrics['confusion_matrix'] = self._confusion_matrix(preds, targets)
        
        return metrics
    
    def _top_k_accuracy(self, probs, targets, k=5):
        """Compute top-k accuracy."""
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        correct = np.array([t in top_k_preds[i] for i, t in enumerate(targets)])
        return correct.mean() * 100
    
    def _confusion_matrix(self, preds, targets):
        """Compute confusion matrix."""
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
        for p, t in zip(preds, targets):
            cm[t, p] += 1
        return cm.tolist()
    
    def measure_latency(self, input_size=(1, 3, 224, 224), num_runs=100, warmup=10):
        """Measure inference latency."""
        dummy_input = torch.randn(input_size).to(self.device)
        
        for _ in range(warmup):
            _ = self.model(dummy_input)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(num_runs):
            _ = self.model(dummy_input)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        total_time = time.time() - start
        avg_latency = (total_time / num_runs) * 1000
        
        return {
            'avg_latency_ms': avg_latency,
            'throughput_fps': 1000 / avg_latency,
            'num_runs': num_runs
        }
    
    def count_parameters(self):
        """Count model parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            'total_params': total,
            'trainable_params': trainable,
            'total_params_millions': total / 1e6,
        }
    
    def estimate_flops(self, input_size=(1, 3, 224, 224)):
        """Estimate FLOPs using thop library if available."""
        try:
            from thop import profile, clever_format
            dummy_input = torch.randn(input_size).to(self.device)
            flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
            flops_str, params_str = clever_format([flops, params], "%.3f")
            return {
                'flops': flops,
                'flops_formatted': flops_str,
                'params_formatted': params_str
            }
        except ImportError:
            return {'error': 'thop library not installed. Run: pip install thop'}


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks.
    
    Generates class activation maps to visualize which regions
    of the input image are important for predictions.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap."""
        self.model.eval()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()


def main(args):
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    val_transform = get_val_transforms(img_size=args.img_size)
    
    if args.dataset == 'cifar10':
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=val_transform, download=True)
        num_classes = 10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    elif args.dataset == 'cifar100':
        val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=val_transform, download=True)
        num_classes = 100
        class_names = None
    elif args.dataset == 'imagenet':
        val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=val_transform)
        num_classes = 1000
        class_names = None
    else:
        val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=val_transform)
        num_classes = len(val_dataset.classes)
        class_names = val_dataset.classes
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    if args.model == 'emn_tiny':
        model = emn_tiny(num_classes=num_classes)
    elif args.model == 'emn_small':
        model = emn_small(num_classes=num_classes)
    elif args.model == 'emn_base':
        model = emn_base(num_classes=num_classes)
    else:
        model = EfficientMicroNet(num_classes=num_classes, variant=args.variant)
    
    if args.checkpoint:
        print(f'Loading checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    evaluator = ModelEvaluator(model, device, num_classes)
    
    print('\n' + '='*60)
    print('MODEL EFFICIENCY METRICS')
    print('='*60)
    
    params = evaluator.count_parameters()
    print(f"Total Parameters: {params['total_params']:,} ({params['total_params_millions']:.2f}M)")
    
    latency = evaluator.measure_latency(input_size=(1, 3, args.img_size, args.img_size))
    print(f"Inference Latency: {latency['avg_latency_ms']:.2f} ms")
    print(f"Throughput: {latency['throughput_fps']:.1f} FPS")
    
    flops = evaluator.estimate_flops(input_size=(1, 3, args.img_size, args.img_size))
    if 'error' not in flops:
        print(f"FLOPs: {flops['flops_formatted']}")
    
    print('\n' + '='*60)
    print('ACCURACY METRICS')
    print('='*60)
    
    metrics = evaluator.evaluate(val_loader)
    
    print(f"Top-1 Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
    print(f"Mean Class Accuracy: {metrics['mean_class_accuracy']:.2f}%")
    print(f"Macro Precision: {metrics['macro_precision']:.2f}%")
    print(f"Macro Recall: {metrics['macro_recall']:.2f}%")
    print(f"Macro F1-Score: {metrics['macro_f1']:.2f}%")
    
    if class_names and len(class_names) <= 20:
        print('\n' + '-'*40)
        print('Per-Class Accuracy:')
        for i, (name, acc) in enumerate(zip(class_names, metrics['per_class_accuracy'])):
            print(f"  {name}: {acc:.2f}%")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model': args.model,
        'dataset': args.dataset,
        'parameters': params,
        'latency': latency,
        'accuracy': metrics['accuracy'],
        'top5_accuracy': metrics['top5_accuracy'],
        'mean_class_accuracy': metrics['mean_class_accuracy'],
        'macro_precision': metrics['macro_precision'],
        'macro_recall': metrics['macro_recall'],
        'macro_f1': metrics['macro_f1'],
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nResults saved to: {output_dir / "evaluation_results.json"}')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate EfficientMicroNet')
    
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet', 'custom'])
    parser.add_argument('--output-dir', type=str, default='./eval_results')
    
    parser.add_argument('--model', type=str, default='emn_small',
                        choices=['emn_tiny', 'emn_small', 'emn_base', 'custom'])
    parser.add_argument('--variant', type=str, default='small')
    parser.add_argument('--checkpoint', type=str, default='')
    
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
