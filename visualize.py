"""
Visualization utilities for EfficientMicroNet.

Includes:
- Training curves plotting
- Confusion matrix heatmap
- Grad-CAM visualizations
- Model architecture visualization
- Feature map visualization
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


def plot_training_curves(history_path, output_dir='./plots'):
    """Plot training and validation curves from history file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Training curves saved to: {output_dir / "training_curves.png"}')


def plot_confusion_matrix(confusion_matrix, class_names=None, output_dir='./plots', normalize=True):
    """Plot confusion matrix heatmap."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm = np.array(confusion_matrix)
    
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    num_classes = len(cm)
    figsize = max(8, num_classes * 0.5)
    
    plt.figure(figsize=(figsize, figsize))
    
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    sns.heatmap(
        cm, annot=num_classes <= 20, fmt='.2f' if normalize else 'd',
        cmap='Blues', xticklabels=class_names, yticklabels=class_names,
        square=True, cbar_kws={'shrink': 0.8}
    )
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    
    filename = 'confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png'
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Confusion matrix saved to: {output_dir / filename}')


def visualize_gradcam(model, image_path, target_layer, class_idx=None, 
                      output_dir='./plots', device='cuda'):
    """Generate and visualize Grad-CAM heatmap."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    gradients = None
    activations = None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()
    
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    
    model.eval()
    output = model(input_tensor)
    
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
    
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1
    output.backward(gradient=one_hot)
    
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    handle_f.remove()
    handle_b.remove()
    
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    heatmap = plt.cm.jet(cam)[:, :, :3]
    overlay = 0.5 * img_array + 0.5 * heatmap
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay (Class: {class_idx})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradcam_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Grad-CAM visualization saved to: {output_dir / "gradcam_visualization.png"}')
    
    return cam


def visualize_feature_maps(model, image_path, layer_name, output_dir='./plots', 
                           device='cuda', num_features=16):
    """Visualize feature maps from a specific layer."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    feature_maps = None
    
    def hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output.detach()
    
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook)
            break
    
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    
    handle.remove()
    
    if feature_maps is None:
        print(f"Layer '{layer_name}' not found!")
        return
    
    features = feature_maps.squeeze().cpu().numpy()
    num_channels = min(num_features, features.shape[0])
    
    grid_size = int(np.ceil(np.sqrt(num_channels)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    for i in range(grid_size * grid_size):
        ax = axes[i // grid_size, i % grid_size]
        if i < num_channels:
            ax.imshow(features[i], cmap='viridis')
            ax.set_title(f'Channel {i}')
        ax.axis('off')
    
    plt.suptitle(f'Feature Maps from {layer_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'feature_maps_{layer_name.replace(".", "_")}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Feature maps saved to: {output_dir}')


def plot_model_comparison(results_list, output_dir='./plots'):
    """Plot comparison of multiple models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = [r['model'] for r in results_list]
    accuracies = [r['accuracy'] for r in results_list]
    params = [r['parameters']['total_params_millions'] for r in results_list]
    latencies = [r['latency']['avg_latency_ms'] for r in results_list]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    
    bars1 = axes[0].bar(models, accuracies, color=colors)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    for bar, acc in zip(bars1, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    bars2 = axes[1].bar(models, params, color=colors)
    axes[1].set_ylabel('Parameters (M)')
    axes[1].set_title('Model Size Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    for bar, p in zip(bars2, params):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{p:.2f}M', ha='center', va='bottom', fontsize=9)
    
    bars3 = axes[2].bar(models, latencies, color=colors)
    axes[2].set_ylabel('Latency (ms)')
    axes[2].set_title('Inference Latency Comparison')
    axes[2].tick_params(axis='x', rotation=45)
    for bar, lat in zip(bars3, latencies):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{lat:.1f}ms', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Model comparison saved to: {output_dir / "model_comparison.png"}')


def plot_accuracy_vs_params(results_list, output_dir='./plots'):
    """Plot accuracy vs parameters scatter plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    for r in results_list:
        plt.scatter(
            r['parameters']['total_params_millions'],
            r['accuracy'],
            s=100,
            label=r['model']
        )
        plt.annotate(
            r['model'],
            (r['parameters']['total_params_millions'], r['accuracy']),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9
        )
    
    plt.xlabel('Parameters (Millions)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Model Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_params.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Accuracy vs params plot saved to: {output_dir / "accuracy_vs_params.png"}')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualization utilities')
    parser.add_argument('--history', type=str, help='Path to history.json')
    parser.add_argument('--output-dir', type=str, default='./plots')
    
    args = parser.parse_args()
    
    if args.history:
        plot_training_curves(args.history, args.output_dir)
