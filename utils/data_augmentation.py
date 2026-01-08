"""
Advanced Data Augmentation Strategies for EfficientMicroNet.

Includes:
1. AutoAugment - Learned augmentation policies
2. RandAugment - Random augmentation with magnitude
3. CutMix and MixUp - Advanced mixing strategies
4. Progressive Resizing - Curriculum learning approach
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
import random


class RandAugment:
    """
    RandAugment: Practical automated data augmentation.
    
    Randomly selects N transformations from a set of K augmentations
    and applies them with magnitude M.
    """
    def __init__(self, n=2, m=9):
        self.n = n
        self.m = m
        self.augment_list = [
            ('Identity', 0, 1),
            ('AutoContrast', 0, 1),
            ('Equalize', 0, 1),
            ('Rotate', -30, 30),
            ('Solarize', 0, 256),
            ('Color', 0.1, 1.9),
            ('Posterize', 4, 8),
            ('Contrast', 0.1, 1.9),
            ('Brightness', 0.1, 1.9),
            ('Sharpness', 0.1, 1.9),
            ('ShearX', -0.3, 0.3),
            ('ShearY', -0.3, 0.3),
            ('TranslateX', -0.3, 0.3),
            ('TranslateY', -0.3, 0.3),
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op_name, min_val, max_val in ops:
            val = min_val + (max_val - min_val) * (self.m / 30)
            img = self._apply_op(img, op_name, val)
        return img

    def _apply_op(self, img, op_name, val):
        if op_name == 'Identity':
            return img
        elif op_name == 'AutoContrast':
            return ImageOps.autocontrast(img)
        elif op_name == 'Equalize':
            return ImageOps.equalize(img)
        elif op_name == 'Rotate':
            return img.rotate(val)
        elif op_name == 'Solarize':
            return ImageOps.solarize(img, int(val))
        elif op_name == 'Color':
            return ImageEnhance.Color(img).enhance(val)
        elif op_name == 'Posterize':
            return ImageOps.posterize(img, int(val))
        elif op_name == 'Contrast':
            return ImageEnhance.Contrast(img).enhance(val)
        elif op_name == 'Brightness':
            return ImageEnhance.Brightness(img).enhance(val)
        elif op_name == 'Sharpness':
            return ImageEnhance.Sharpness(img).enhance(val)
        elif op_name == 'ShearX':
            return img.transform(img.size, Image.AFFINE, (1, val, 0, 0, 1, 0))
        elif op_name == 'ShearY':
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, val, 1, 0))
        elif op_name == 'TranslateX':
            return img.transform(img.size, Image.AFFINE, (1, 0, val * img.size[0], 0, 1, 0))
        elif op_name == 'TranslateY':
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, val * img.size[1]))
        return img


class CutMix:
    """
    CutMix: Cut and paste patches between training images.
    
    Improves localization ability and prevents the model from
    focusing only on a small discriminative part of objects.
    """
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch, targets):
        if random.random() > self.prob:
            return batch, targets, targets, 1.0

        batch_size = batch.size(0)
        indices = torch.randperm(batch_size)
        
        lam = np.random.beta(self.alpha, self.alpha)
        
        _, _, H, W = batch.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        batch[:, :, bby1:bby2, bbx1:bbx2] = batch[indices, :, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return batch, targets, targets[indices], lam


class MixUp:
    """
    MixUp: Convex combinations of pairs of examples and their labels.
    
    Creates virtual training examples by interpolating between
    pairs of examples and their labels.
    """
    def __init__(self, alpha=0.2, prob=0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch, targets):
        if random.random() > self.prob:
            return batch, targets, targets, 1.0

        batch_size = batch.size(0)
        indices = torch.randperm(batch_size)
        
        lam = np.random.beta(self.alpha, self.alpha)
        
        mixed_batch = lam * batch + (1 - lam) * batch[indices]
        
        return mixed_batch, targets, targets[indices], lam


class CutMixMixUp:
    """
    Combined CutMix and MixUp augmentation.
    Randomly chooses between CutMix and MixUp for each batch.
    """
    def __init__(self, cutmix_alpha=1.0, mixup_alpha=0.2, prob=0.5, switch_prob=0.5):
        self.cutmix = CutMix(alpha=cutmix_alpha, prob=1.0)
        self.mixup = MixUp(alpha=mixup_alpha, prob=1.0)
        self.prob = prob
        self.switch_prob = switch_prob

    def __call__(self, batch, targets):
        if random.random() > self.prob:
            return batch, targets, targets, 1.0
        
        if random.random() < self.switch_prob:
            return self.cutmix(batch, targets)
        else:
            return self.mixup(batch, targets)


def get_train_transforms(img_size=224, auto_augment=True, randaugment_n=2, randaugment_m=9):
    """
    Get training transforms with optional augmentation strategies.
    """
    transform_list = [
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    
    if auto_augment:
        transform_list.append(RandAugment(n=randaugment_n, m=randaugment_m))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33)),
    ])
    
    return transforms.Compose(transform_list)


def get_val_transforms(img_size=224, crop_pct=0.875):
    """
    Get validation/test transforms.
    """
    resize_size = int(img_size / crop_pct)
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ProgressiveResizing:
    """
    Progressive Resizing: Start with smaller images and gradually increase.
    
    This curriculum learning approach helps the model learn coarse features
    first and then fine-grained details, improving both training speed and accuracy.
    """
    def __init__(self, start_size=128, end_size=224, num_epochs=100):
        self.start_size = start_size
        self.end_size = end_size
        self.num_epochs = num_epochs

    def get_size(self, epoch):
        progress = min(epoch / (self.num_epochs * 0.7), 1.0)
        current_size = int(self.start_size + (self.end_size - self.start_size) * progress)
        current_size = (current_size // 32) * 32
        return max(current_size, self.start_size)

    def get_transforms(self, epoch, is_train=True):
        img_size = self.get_size(epoch)
        if is_train:
            return get_train_transforms(img_size)
        return get_val_transforms(img_size)
