#!/usr/bin/env python3
"""
Inference script for trained multi-label jewelry classifier
HARDCODED PATHS VERSION - Update paths below
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import timm

# ============================================================
# HARDCODED PATHS - UPDATE THESE
# ============================================================
CHECKPOINT_PATH = r"C:\Users\Crown Tech\jupyter\raresenc\microservice1\best_model.pth"
TEST_IMAGE_PATH = r"C:\Users\Crown Tech\jupyter\raresenc\microservice1\human_image_35.jpg"  # Single image
TEST_FOLDER_PATH = r"C:\Users\Crown Tech\jupyter\raresenc\microservice1\test"  # Folder of images
THRESHOLD = 0.55
DEVICE = "cpu"  # or "cpu"
# ============================================================

# Add this class to fix the unpickling error
class TrainingConfig:
    """Training configuration class - needed for loading checkpoint"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class LabelEncoder:
    """Encoder for multi-label classification"""
    
    def __init__(self, all_labels):
        self.classes = sorted(list(set(all_labels)))
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
    def labels_to_vector(self, labels):
        """Convert list of labels to binary vector"""
        import torch
        vector = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in labels:
            if label in self.class_to_idx:
                vector[self.class_to_idx[label]] = 1.0
        return vector

class MultiLabelViT(nn.Module):
    """Vision Transformer for multi-label classification"""
    
    def __init__(self, model_name, num_classes, img_size=(384, 384), dropout=0.2):
        super().__init__()
        self.img_size = img_size
        
        if 'swin' in model_name:
            if isinstance(img_size, int):
                swin_img_size = (img_size, img_size)
            elif isinstance(img_size, tuple):
                swin_img_size = (img_size[0], img_size[0]) if img_size[0] == img_size[1] else img_size
            else:
                swin_img_size = (384, 384)
            
            self.backbone = timm.create_model(
                model_name, pretrained=False, num_classes=0, img_size=swin_img_size
            )
        else:
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        
        if hasattr(self.backbone, "num_features"):
            self.feat_dim = self.backbone.num_features
        else:
            self.feat_dim = 768
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes * 4),
            nn.GELU(),
            nn.LayerNorm(num_classes * 4),
            nn.Dropout(dropout / 2),
            nn.Linear(num_classes * 4, num_classes * 2),
            nn.GELU(),
            nn.Dropout(dropout / 4),
            nn.Linear(num_classes * 2, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        if len(features.shape) == 3:
            features = features.mean(dim=1)
        elif len(features.shape) == 4:
            features = features.mean(dim=[2, 3])
        
        logits = self.classifier(features)
        return logits

class ScaleAndPadToSquare:
    """Scale and pad image to square"""
    def __init__(self, target_size=384, fill_color=(0, 0, 0)):
        self.target_size = target_size
        self.fill_color = fill_color
    
    def __call__(self, img):
        w, h = img.size
        scale = min(self.target_size / w, self.target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        square_img = Image.new('RGB', (self.target_size, self.target_size), self.fill_color)
        
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        square_img.paste(img_resized, (paste_x, paste_y))
        
        return square_img

def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load checkpoint with weights_only=False to allow custom classes
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config
    label_encoder = checkpoint['label_encoder']
    num_classes = label_encoder.num_classes
    
    # Reconstruct model
    model = MultiLabelViT(
        model_name='swin_base_patch4_window12_384',
        num_classes=num_classes,
        img_size=(384, 384),
        dropout=0.2
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Classes: {label_encoder.classes}")
    print(f"Best mAP: {checkpoint['metrics']['mAP']:.4f}")
    
    return model, label_encoder

def get_transforms():
    """Get inference transforms"""
    return transforms.Compose([
        ScaleAndPadToSquare(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_image(image_path, model, label_encoder, transform, device='cuda', threshold=0.5):
    """Predict jewelry types in an image"""
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Get predictions above threshold
    predictions = []
    for idx, prob in enumerate(probs):
        if prob > threshold:
            predictions.append({
                'class': label_encoder.idx_to_class[idx],
                'confidence': float(prob)
            })
    
    # Sort by confidence
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    return predictions, probs

def main():
    """Main function with hardcoded paths"""
    
    # Check if checkpoint exists
    if not Path(CHECKPOINT_PATH).exists():
        print(f"Error: Checkpoint not found: {CHECKPOINT_PATH}")
        print("Update CHECKPOINT_PATH at the top of this script")
        return
    
    # Load model
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, label_encoder = load_model(CHECKPOINT_PATH, device)
    transform = get_transforms()
    
    # Single image inference
    if Path(TEST_IMAGE_PATH).exists():
        print(f"\nAnalyzing single image: {TEST_IMAGE_PATH}")
        predictions, all_probs = predict_image(TEST_IMAGE_PATH, model, label_encoder, transform, device, THRESHOLD)
        
        print("\n" + "="*60)
        print("PREDICTIONS")
        print("="*60)
        
        if predictions:
            for pred in predictions:
                print(f"  {pred['class']:<20} {pred['confidence']:.2%}")
        else:
            print("  No jewelry detected above threshold")
        
        print("\n" + "="*60)
        print("ALL CLASS PROBABILITIES")
        print("="*60)
        for idx, prob in enumerate(all_probs):
            class_name = label_encoder.idx_to_class[idx]
            print(f"  {class_name:<20} {prob:.2%}")
    else:
        print(f"\nWarning: Single test image not found: {TEST_IMAGE_PATH}")
    
    # Batch processing folder
    print(f"\n\nBatch processing folder: {TEST_FOLDER_PATH}")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    
    if Path(TEST_FOLDER_PATH).exists():
        for ext in image_extensions:
            image_files.extend(Path(TEST_FOLDER_PATH).glob(ext))
        
        if image_files:
            print(f"Found {len(image_files)} images")
            print("\n" + "="*60)
            print("BATCH RESULTS")
            print("="*60)
            
            for img_path in sorted(image_files):
                try:
                    predictions, _ = predict_image(str(img_path), model, label_encoder, transform, device, THRESHOLD)
                    pred_labels = [f"{p['class']}({p['confidence']:.0%})" for p in predictions]
                    if pred_labels:
                        print(f"{img_path.name:<40} -> {', '.join(pred_labels)}")
                    else:
                        print(f"{img_path.name:<40} -> No jewelry detected")
                except Exception as e:
                    print(f"{img_path.name:<40} -> Error: {str(e)}")
        else:
            print(f"No images found in {TEST_FOLDER_PATH}")
    else:
        print(f"Folder not found: {TEST_FOLDER_PATH}")
        print("Update TEST_FOLDER_PATH at the top of this script")

if __name__ == "__main__":
    main()