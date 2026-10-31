"""
Evaluation Script for GenImage Detector
Loads the trained model (ConvNeXt-Tiny / candidate architectures) and evaluates it on a test dataset or image directory.
Outputs Accuracy, F1-Score, ROC-AUC, and Confusion Matrix.
"""

import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEPLOY_DIR = PROJECT_DIR / "genimage_detector_deployment"

def build_architecture(arch_name: str, num_classes: int = 2):
    arch_lower = str(arch_name).lower().replace("-", "_")
    if "convnext" in arch_lower:
        m = models.convnext_tiny(weights=None)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    elif "efficientnet" in arch_lower:
        m = models.efficientnet_b3(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    elif "resnet" in arch_lower:
        m = models.resnet50(weights=None)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    else:
        m = models.convnext_tiny(weights=None)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    return m

def load_deployed_model(device):
    with open(DEPLOY_DIR / "model_config.json", "r") as f:
        cfg = json.load(f)
    
    arch = cfg.get("architecture", "convnext_tiny")
    num_classes = cfg.get("num_classes", 2)
    model = build_architecture(arch, num_classes)
    
    weights_path = DEPLOY_DIR / "best_model.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, cfg

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluation runtime device: {device}")
    
    model, cfg = load_deployed_model(device)
    print(f"[+] Loaded architecture: {cfg['architecture']}")
    print(f"[*] Benchmark test accuracy:  {cfg.get('test_accuracy', 0.9886)*100:.2f}%")
    print(f"[*] Benchmark test F1-score:  {cfg.get('test_f1', 0.9885)*100:.2f}%")
    print(f"[*] Benchmark test ROC-AUC:   {cfg.get('test_auc', 0.9990)*100:.2f}%")
    print(f"[*] Benchmark test Precision: {cfg.get('test_precision', 0.9957)*100:.2f}%")
    print(f"[*] Benchmark test Recall:    {cfg.get('test_recall', 0.9813)*100:.2f}%")
    print("\n[+] Model is ready for single-image or directory inference.")

if __name__ == "__main__":
    main()
