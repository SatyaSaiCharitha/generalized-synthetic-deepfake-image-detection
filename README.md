# GenImage Detector — Generalized AI & Deepfake Image Detection
**Academic Research & Production Deployment System**

This repository provides a production-ready, professional web interface and REST API for detecting AI-generated and synthetic imagery across multi-domain datasets.

---

## 🔬 Model & Benchmark Overview

- **Core Task**: Binary Classification — `REAL` vs `AI-GENERATED`
- **Benchmark Dataset**: [Tiny-GenImage](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage) (Multi-domain subset spanning ImageNet categories and 7 generative architectures)
- **Generators Evaluated**: Stable Diffusion, Midjourney, BigGAN, GLIDE, ADM, VQDM, Wukong
- **Deployed Architecture**: **ConvNeXt-Tiny** (28.6M parameters)
  - Selected over **ResNet-50** and **EfficientNet-B3** based on lowest validation loss (0.2182) and superior cross-domain generalization.
- **Test Performance on Held-Out Split (5,251 Samples)**:
  - **Test Accuracy**: **98.86%**
  - **F1-Score**: **98.85%**
  - **ROC-AUC**: **99.90%**
  - **AI Precision**: **99.57%**
  - **AI Recall**: **98.13%**

---

## 🚀 Quick Start (Running the UI)

### Option 1: Double-Click Launcher (Windows)
Double-click [`run.bat`](run.bat).

### Option 2: Command Line
```powershell
# From the project directory:
python run_backend.py
```
Open your browser to: **[http://localhost:8000](http://localhost:8000)**

---

## 🎨 UI Features

1. **Detection Studio**:
   - **One-Click Presets**: Test pre-loaded AI and Real samples immediately with a single click.
   - **Drag-and-Drop Upload**: Upload any JPEG, PNG, WEBP, or BMP image. Clipboard paste (Ctrl+V) supported.
   - **Verdict Banner**: Glowing Real vs. AI-Generated verdict indicator with confidence scoring.
   - **Synthetic Risk Assessment**: Categorizes risk levels (Severe, High, Inconclusive, Authentic).
   - **Probability Distribution Gauges**: Animated percentage meters for both classes.
   - **Artifact & Spectral Inspection Stage**: Toggle between the original image and a high-pass frequency residual map (Laplacian filter) highlighting generative boundary anomalies and smoothing discrepancies.
   - **Inference Diagnostics**: Real-time latency tracking (typically 30–50ms), image resolution, and hardware device.

2. **Model Comparison & Benchmark Tab**:
   - Side-by-side comparison table of **ResNet-50**, **ConvNeXt-Tiny**, and **EfficientNet-B3**.
   - Generator-wise detection breakdown matrix (BigGAN: 100%, VQDM: 99.7%, Real: 99.5%, GLIDE: 98.7%, Wukong: 97.9%, Midjourney: 96.9%, ADM: 95.1%).
   - Full 2x2 confusion matrix (N = 5,251).

3. **Methodology & Dataset Tab**:
   - Step-by-step modular pipeline diagram (Dataset → Preprocessing → Backbone Selection → Classification & Forensics).
   - Academic explanations of 7x7 depthwise convolutions, inverted bottlenecks, and cross-domain robustness.

---

## 🔌 API Endpoints

- `GET /` — Serves the interactive web interface.
- `GET /api/health` — Health check, execution device (CPU/CUDA), and model status.
- `GET /api/model-info` — Academic benchmark data, candidate model metrics, and generator accuracy.
- `GET /api/samples` — List of demonstration preset samples.
- `POST /api/predict` — Image classification endpoint. Accepts either `file` (multipart upload) or `preset_sample` (string ID). Returns classification probabilities, verdict, confidence, risk assessment, and base64 artifact heatmap.

---

## 📁 Project Structure

```
project/
├── genimage_detector_deployment/    # Deployed model weights & configs
│   ├── best_model.pth              # ConvNeXt-Tiny PyTorch state dict (98.86% Acc)
│   ├── model_config.json           # Model metadata & test scores
│   ├── normalization_config.json   # Mean & std normalization constants
│   └── class_names.json            # Class labels: REAL, AI-GENERATED
├── frontend/                        # Web application assets
│   ├── index.html                  # Semantic UI structure
│   ├── style.css                   # Glassmorphic dark theme stylesheet
│   ├── app.js                      # UI controller & async API handler
│   └── samples/                    # Built-in evaluation test images
├── training/                        # Offline training and evaluation scripts
│   ├── inspect_dataset.py          # Dataset structure and generator counter
│   └── evaluate.py                 # Evaluation benchmark runner
├── train_tiny_genimage_local.ipynb # Training notebook (3 candidate models)
├── server.py                       # FastAPI production backend
├── run_backend.py                  # Automatic launcher script
├── run.bat                         # Windows batch launcher
└── requirements.txt                # Python dependencies
```
Note: The trained model weights (best_model.pth) are downloaded separately from Hugging Face and are not stored in this GitHub repository.

## 📥 Download Trained Model

Download the trained ConvNeXt-Tiny model weights here:

👉 [Download best_model.pth](https://huggingface.co/SaiCharitha1710/generalized-synthetic-deepfake-detector/resolve/main/best_model.pth)

After downloading, place the file inside the `genimage_detector_deployment/` folder.
