"""
Inference Script for EfficientMicroNet.

Provides easy-to-use inference functions for:
- Single image prediction
- Batch prediction
- Real-time webcam inference
- ONNX export for deployment
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from models import emn_tiny, emn_small, emn_base, EfficientMicroNet


class EMNPredictor:
    """
    Easy-to-use predictor class for EfficientMicroNet.
    """
    
    def __init__(self, checkpoint_path, model_type='emn_small', num_classes=10, 
                 class_names=None, device=None):
        """
        Initialize predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_type: Model variant ('emn_tiny', 'emn_small', 'emn_base')
            num_classes: Number of output classes
            class_names: Optional list of class names
            device: Device to run inference on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = num_classes
        
        if model_type == 'emn_tiny':
            self.model = emn_tiny(num_classes=num_classes)
        elif model_type == 'emn_small':
            self.model = emn_small(num_classes=num_classes)
        elif model_type == 'emn_base':
            self.model = emn_base(num_classes=num_classes)
        else:
            self.model = EfficientMicroNet(num_classes=num_classes)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def predict(self, image, top_k=5):
        """
        Predict class for a single image.
        
        Args:
            image: PIL Image, numpy array, or path to image
            top_k: Number of top predictions to return
            
        Returns:
            dict with predictions, probabilities, and class names
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        top_probs, top_indices = probs.topk(min(top_k, self.num_classes), dim=1)
        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()
        
        if top_k == 1:
            top_probs = [top_probs.item()]
            top_indices = [top_indices.item()]
        
        results = {
            'predicted_class': int(top_indices[0]),
            'confidence': float(top_probs[0]),
            'top_k_classes': [int(idx) for idx in top_indices],
            'top_k_probs': [float(p) for p in top_probs],
        }
        
        if self.class_names:
            results['predicted_name'] = self.class_names[top_indices[0]]
            results['top_k_names'] = [self.class_names[idx] for idx in top_indices]
        
        return results
    
    @torch.no_grad()
    def predict_batch(self, images):
        """
        Predict classes for a batch of images.
        
        Args:
            images: List of PIL Images, numpy arrays, or paths
            
        Returns:
            List of prediction dicts
        """
        batch_tensors = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img).convert('RGB')
            batch_tensors.append(self.transform(img))
        
        batch = torch.stack(batch_tensors).to(self.device)
        
        outputs = self.model(batch)
        probs = F.softmax(outputs, dim=1)
        
        predictions = []
        for i in range(len(images)):
            pred_class = probs[i].argmax().item()
            confidence = probs[i].max().item()
            
            result = {
                'predicted_class': pred_class,
                'confidence': confidence,
            }
            
            if self.class_names:
                result['predicted_name'] = self.class_names[pred_class]
            
            predictions.append(result)
        
        return predictions
    
    def export_onnx(self, output_path, input_size=(1, 3, 224, 224)):
        """
        Export model to ONNX format for deployment.
        
        Args:
            output_path: Path to save ONNX model
            input_size: Input tensor size
        """
        dummy_input = torch.randn(input_size).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f'Model exported to: {output_path}')
    
    def export_torchscript(self, output_path):
        """
        Export model to TorchScript for deployment.
        
        Args:
            output_path: Path to save TorchScript model
        """
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        traced_model = torch.jit.trace(self.model, dummy_input)
        traced_model.save(output_path)
        print(f'TorchScript model saved to: {output_path}')


def main(args):
    """Main inference function."""
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    class_names = cifar10_classes if args.dataset == 'cifar10' else None
    num_classes = 10 if args.dataset == 'cifar10' else args.num_classes
    
    predictor = EMNPredictor(
        checkpoint_path=args.checkpoint,
        model_type=args.model,
        num_classes=num_classes,
        class_names=class_names
    )
    
    if args.export_onnx:
        predictor.export_onnx(args.export_onnx)
        return
    
    if args.export_torchscript:
        predictor.export_torchscript(args.export_torchscript)
        return
    
    if args.image:
        print(f'\nPredicting: {args.image}')
        result = predictor.predict(args.image, top_k=5)
        
        print(f"\nPredicted Class: {result['predicted_class']}", end='')
        if 'predicted_name' in result:
            print(f" ({result['predicted_name']})")
        else:
            print()
        print(f"Confidence: {result['confidence']*100:.2f}%")
        
        print("\nTop-5 Predictions:")
        for i, (cls, prob) in enumerate(zip(result['top_k_classes'], result['top_k_probs'])):
            name = result.get('top_k_names', [''] * 5)[i]
            print(f"  {i+1}. Class {cls} {f'({name})' if name else ''}: {prob*100:.2f}%")
    
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        print(f'\nProcessing {len(image_files)} images from {image_dir}')
        
        results = predictor.predict_batch([str(f) for f in image_files])
        
        for img_path, result in zip(image_files, results):
            name = result.get('predicted_name', str(result['predicted_class']))
            print(f"{img_path.name}: {name} ({result['confidence']*100:.1f}%)")


def parse_args():
    parser = argparse.ArgumentParser(description='EfficientMicroNet Inference')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='emn_small',
                        choices=['emn_tiny', 'emn_small', 'emn_base'])
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num-classes', type=int, default=10)
    
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image-dir', type=str, help='Path to directory of images')
    
    parser.add_argument('--export-onnx', type=str, help='Export to ONNX format')
    parser.add_argument('--export-torchscript', type=str, help='Export to TorchScript')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
