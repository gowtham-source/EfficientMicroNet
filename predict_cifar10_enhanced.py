#!/usr/bin/env python3
"""
Enhanced CIFAR-10 prediction with original and processed images comparison.
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


class EnhancedCIFAR10Predictor:
    def __init__(self, checkpoint_path, model_type='emn_small', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # CIFAR-10 classes
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Transforms for different views
        self.original_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.processed_transform = transforms.Compose([
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
    
    def visualize_predictions(self, num_samples=4, save_path='enhanced_cifar10_predictions.png'):
        """Visualize predictions showing both original and processed images."""
        # Load test dataset
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.original_transform
        )
        
        # Select random samples
        indices = np.random.choice(len(testset), num_samples, replace=False)
        
        # Create figure with larger size
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
        
        for row_idx, sample_idx in enumerate(indices):
            # Get image and true label
            image, true_label = testset[sample_idx]
            
            # Original 32x32 image
            img_np = image.numpy().transpose(1, 2, 0)
            axes[row_idx, 0].imshow(img_np)
            axes[row_idx, 0].set_title(f'Original (32x32)\nTrue: {self.classes[true_label]}', 
                                      fontsize=10)
            axes[row_idx, 0].axis('off')
            
            # Processed 224x224 image (what model sees)
            # Convert tensor back to PIL for transforms
            img_pil = transforms.ToPILImage()(image)
            processed_img = self.processed_transform(img_pil)
            
            # Denormalize for visualization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            processed_np = processed_img.numpy().transpose(1, 2, 0)
            processed_np = std * processed_np + mean
            processed_np = np.clip(processed_np, 0, 1)
            
            axes[row_idx, 1].imshow(processed_np)
            axes[row_idx, 1].set_title(f'Processed (224x224)\nUpscaled & Normalized', 
                                      fontsize=10)
            axes[row_idx, 1].axis('off')
            
            # Prediction visualization
            # Predict
            image_batch = processed_img.unsqueeze(0).to(self.device)
            result = self.predict_batch(image_batch)[0]
            
            # Create prediction visualization
            axes[row_idx, 2].imshow(processed_np)
            axes[row_idx, 2].axis('off')
            
            # Add bounding box
            rect = patches.Rectangle((10, 10), 204, 204, linewidth=3, 
                                   edgecolor='lime' if result['predicted_class'] == self.classes[true_label] else 'red', 
                                   facecolor='none')
            axes[row_idx, 2].add_patch(rect)
            
            # Create prediction text
            pred_text = f"Predicted: {result['predicted_class']}\n"
            pred_text += f"Confidence: {result['confidence']:.1%}\n"
            pred_text += f"Top-3:\n"
            for i, pred in enumerate(result['predictions'][:3]):
                pred_text += f"  {i+1}. {pred['class']}: {pred['confidence']:.1%}\n"
            
            # Add background box for text
            props = dict(boxstyle='round', facecolor='black', alpha=0.8)
            axes[row_idx, 2].text(0.02, 0.98, pred_text, transform=axes[row_idx, 2].transAxes,
                                  fontsize=9, verticalalignment='top', color='white',
                                  bbox=props)
            
            # Color code title based on correctness
            color = 'green' if result['predicted_class'] == self.classes[true_label] else 'red'
            axes[row_idx, 2].set_title(f'Prediction\n{"✓ CORRECT" if result["predicted_class"] == self.classes[true_label] else "✗ WRONG"}', 
                                      fontsize=10, color=color, fontweight='bold')
        
        plt.suptitle(f'CIFAR-10 Predictions using {self.model_type.upper()}\n'
                    f'Device: {self.device} | Showing Original vs Processed Images', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Enhanced visualization saved to: {save_path}")
        return fig


def main():
    parser = argparse.ArgumentParser(description='Enhanced CIFAR-10 prediction visualization')
    parser.add_argument('--checkpoint', type=str, default='outputs/model_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='emn_small',
                       choices=['emn_tiny', 'emn_small', 'emn_base'],
                       help='Model type')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--num-samples', type=int, default=4,
                       help='Number of test samples to visualize')
    parser.add_argument('--save-path', type=str, default='enhanced_cifar10_predictions.png',
                       help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = EnhancedCIFAR10Predictor(
        checkpoint_path=args.checkpoint,
        model_type=args.model,
        device=args.device
    )
    
    print(f"Loaded {args.model} model from {args.checkpoint}")
    print(f"Using device: {predictor.device}")
    print(f"Note: CIFAR-10 images (32x32) are upscaled to 224x224 for the model")
    
    # Visualize predictions
    predictor.visualize_predictions(
        num_samples=args.num_samples,
        save_path=args.save_path
    )


if __name__ == '__main__':
    main()
