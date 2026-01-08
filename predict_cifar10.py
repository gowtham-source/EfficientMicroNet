#!/usr/bin/env python3
"""
Predict CIFAR-10 test images with bounding boxes and predictions.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from pathlib import Path
import argparse

from models import emn_tiny, emn_small, emn_base, EfficientMicroNet


class CIFAR10Predictor:
    def __init__(self, checkpoint_path, model_type='emn_small', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # CIFAR-10 classes
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Transform for CIFAR-10 (matching training)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, checkpoint_path):
        """Load the trained model."""
        if self.model_type == 'emn_small':
            model = emn_small(num_classes=10)
        elif self.model_type == 'emn_tiny':
            model = emn_tiny(num_classes=10)
        elif self.model_type == 'emn_base':
            model = emn_base(num_classes=10)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        return model
    
    def predict_batch(self, images, top_k=3):
        """Predict on a batch of images."""
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            results = []
            for i in range(len(images)):
                probs, indices = torch.topk(probabilities[i], top_k)
                results.append({
                    'predictions': [
                        {
                            'class': self.classes[idx],
                            'confidence': prob.item()
                        }
                        for prob, idx in zip(probs, indices)
                    ],
                    'predicted_class': self.classes[indices[0]],
                    'confidence': probs[0].item()
                })
            return results
    
    def visualize_predictions(self, num_samples=4, save_path='cifar10_predictions.png'):
        """Visualize predictions on CIFAR-10 test set with bounding boxes."""
        # Load test dataset
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.transform
        )
        
        # Select random samples
        indices = np.random.choice(len(testset), num_samples, replace=False)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, ax in enumerate(indices):
            # Get image and true label
            image, true_label = testset[idx]
            
            # Denormalize for visualization (ImageNet stats)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = image.numpy().transpose(1, 2, 0)
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            
            # Predict
            image_batch = image.unsqueeze(0).to(self.device)
            result = self.predict_batch(image_batch)[0]
            
            # Display image
            axes[idx].imshow(img_np)
            axes[idx].axis('off')
            
            # Add bounding box
            rect = patches.Rectangle((2, 2), 28, 28, linewidth=2, 
                                   edgecolor='lime', facecolor='none')
            axes[idx].add_patch(rect)
            
            # Create prediction text
            pred_text = f"Pred: {result['predicted_class']}\n"
            pred_text += f"Conf: {result['confidence']:.2%}\n"
            pred_text += f"True: {self.classes[true_label]}"
            
            # Add background box for text
            props = dict(boxstyle='round', facecolor='black', alpha=0.7)
            axes[idx].text(0.02, 0.98, pred_text, transform=axes[idx].transAxes,
                          fontsize=9, verticalalignment='top', color='white',
                          bbox=props)
            
            # Color code based on correctness
            color = 'lime' if result['predicted_class'] == self.classes[true_label] else 'red'
            rect.set_edgecolor(color)
            
            axes[idx].set_title(f"Sample {idx+1}", fontsize=10, color=color)
        
        plt.suptitle(f"CIFAR-10 Predictions using {self.model_type.upper()}\n"
                    f"Device: {self.device}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to: {save_path}")
        return fig


def main():
    parser = argparse.ArgumentParser(description='Predict CIFAR-10 test images')
    parser.add_argument('--checkpoint', type=str, default='outputs/model_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='emn_small',
                       choices=['emn_tiny', 'emn_small', 'emn_base'],
                       help='Model type')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--num-samples', type=int, default=4,
                       help='Number of test samples to visualize')
    parser.add_argument('--save-path', type=str, default='cifar10_predictions.png',
                       help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = CIFAR10Predictor(
        checkpoint_path=args.checkpoint,
        model_type=args.model,
        device=args.device
    )
    
    print(f"Loaded {args.model} model from {args.checkpoint}")
    print(f"Using device: {predictor.device}")
    
    # Visualize predictions
    predictor.visualize_predictions(
        num_samples=args.num_samples,
        save_path=args.save_path
    )


if __name__ == '__main__':
    main()
