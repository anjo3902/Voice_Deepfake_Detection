# 🎯 Voice Deepfake Detection System

**Real-time AI-powered voice deepfake detection using AASIST neural architecture**

A production-ready web application that detects AI-generated voice deepfakes with high accuracy, supporting both professional studio recordings and consumer-grade microphones.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📂 PROJECT STRUCTURE

```
Voice_Deepfake_Detection/
│
├── 📄 README.md                         ← Project documentation (this file)
├── 🔧 utils.py                          ← Dataset utilities & testing tools
├── 🎓 train_comprehensive.py            ← Comprehensive training script
├── 🌐 serve_https.py                    ← HTTPS frontend server (port 3000)
├── 📄 comprehensive_push.py             ← Git automation script
│
├── 📁 backend/                          ← Flask REST API + Deep Learning Model
│   ├── requirements.txt                 ← Python dependencies
│   ├── models/
│   │   ├── aasist.py                    ← AASIST model architecture (5.4M params)
│   │   ├── app.py                       ← Flask API endpoints
│   │   ├── enhanced_detector.py         ← Enhanced TTS detection wrapper
│   │   └── feature_extractor.py         ← Audio feature extraction
│   ├── checkpoints/                     ← Trained model weights
│   │   ├── zero_download_model.pth      ← Production model (20.82 MB)
│   │   └── best.pth                     ← Base model checkpoint (20.82 MB)
│   └── uploads/                         ← Temporary audio file storage
│
├── 📁 frontend/                         ← React Web Interface
│   └── dist/                            ← Production build
│       ├── index.html                   ← Main HTML file
│       └── assets/                      ← CSS/JS bundles
│
├── 📁 certificates/                     ← SSL certificates for HTTPS
│   ├── cert.pem                         ← SSL certificate
│   └── key.pem                          ← Private key
│
└── 📁 data/                             ← Training datasets (not in repo)
    ├── ASVspoof2019/                    ← ASVspoof 2019 LA dataset
    ├── downloaded_dataset/              ← LibriSpeech test-clean
    └── your_voice_samples/              ← Custom voice recordings
```

---

## 🚀 QUICK START

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended) or CPU
- Windows/Linux/macOS

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/anjo3902/Voice_Deepfake_Detection.git
cd Voice_Deepfake_Detection
```

2. **Create virtual environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r backend/requirements.txt
```

### Running the Application

#### Option 1: Run Backend and Frontend Separately

**Terminal 1 - Backend API (Port 5000):**
```bash
cd backend/models
python app.py
```

**Terminal 2 - Frontend Server (Port 3000):**
```bash
python serve_https.py
```

Then open your browser:
- **Frontend**: https://localhost:3000
- **Backend API**: https://localhost:5000

> **Note**: You'll see a security warning about self-signed SSL certificates. Click "Advanced" → "Proceed to localhost" to continue.

#### Option 2: Quick Test via API

```python
import requests

# Test the API
response = requests.post(
    'https://localhost:5000/predict',
    files={'file': open('test_audio.wav', 'rb')},
    verify=False  # Skip SSL verification for self-signed cert
)

print(response.json())
# Output: {"is_fake": false, "confidence": 0.9234, "prediction": "REAL"}
```

---

## 📊 KEY FEATURES

### ✨ Core Capabilities

- **🎯 High Accuracy**: 89.59% detection accuracy on diverse voice samples
- **⚡ Real-Time Processing**: ~76.8ms inference time per 4-second audio clip
- **🎤 Consumer Microphone Support**: Works with laptops, USB mics, and phone recordings
- **🌐 Web Interface**: User-friendly React-based frontend with drag-and-drop upload
- **🔒 Secure HTTPS**: SSL-enabled backend and frontend servers
- **🔄 RESTful API**: Easy integration with other applications
- **📊 Confidence Scoring**: Detailed prediction confidence levels
- **🎵 Multiple Format Support**: WAV, FLAC, MP3, M4A, OGG

### 🧠 Technical Features

- **AASIST Architecture**: State-of-the-art neural network (5.4M parameters)
- **CUDA Acceleration**: GPU-accelerated inference on NVIDIA GPUs
- **Robust Feature Extraction**: LFCC, Spectral, and Sinc-based features
- **Batch Processing**: Efficient handling of multiple audio files
- **Comprehensive Training**: Trained on ASVspoof2019 + LibriSpeech + custom datasets

---

## 🎓 MODEL DETAILS

### Architecture: AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal graph attention)

| Parameter | Value |
|-----------|-------|
| **Model Type** | Graph Attention Network + ResNet |
| **Total Parameters** | 5.4 Million |
| **Input** | Raw audio waveform (16kHz, mono) |
| **Output** | Binary classification (Real/Fake) + Confidence |
| **Inference Time** | 14.8ms per 4-second clip (GPU) |
| **Model Size** | 20.82 MB |
| **Framework** | PyTorch 2.7+ |

### Training Details

- **Primary Dataset**: ASVspoof 2019 Logical Access (LA) subset
- **Supplementary Data**: LibriSpeech test-clean, custom recordings
- **Training Strategy**: Comprehensive training to prevent catastrophic forgetting
- **Optimizer**: Adam with learning rate 0.0001
- **Loss Function**: Cross-Entropy Loss
- **Batch Size**: 16
- **Hardware**: NVIDIA RTX 2050 4GB VRAM

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 89.59% |
| **Validation Accuracy** | 92.3% |
| **False Positive Rate** | ~8.5% |
| **Inference Latency** | 76.8ms |

---

## 🔧 TRAINING (OPTIONAL)

The repository includes pre-trained models, but you can retrain if needed:

### 1. Download Training Datasets

```bash
# Download LibriSpeech test-clean subset (~350MB)
python utils.py download
```

### 2. Prepare ASVspoof2019 Dataset

Download ASVspoof2019 LA dataset from [official source](https://datashare.ed.ac.uk/handle/10283/3336) and extract to:
```
data/ASVspoof2019/LA/
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
└── ASVspoof2019_LA_cm_protocols/
```

### 3. Train Comprehensive Model

```bash
python train_comprehensive.py
```

This trains a comprehensive model that includes:
- ✅ ASVspoof traditional TTS spoofs
- ✅ Modern neural TTS (ElevenLabs, modern systems)
- ✅ Real voices from LibriSpeech
- ✅ Custom voice recordings

**Training Output**: New checkpoint saved to `backend/checkpoints/`

---

## 🧪 TESTING & UTILITIES

The `utils.py` script provides several helpful utilities:

```bash
# Download LibriSpeech test dataset
python utils.py download

# Test model on diverse speakers
python utils.py test

# Generate performance graphs
python utils.py graphs

# Run all utilities
python utils.py all

# Show help
python utils.py help
```

---

## 📞 QUICK REFERENCE

| Task | Command |
|------|---------|
| **Install dependencies** | `pip install -r backend/requirements.txt` |
| **Run backend API** | `cd backend/models && python app.py` |
| **Run frontend server** | `python serve_https.py` |
| **Download datasets** | `python utils.py download` |
| **Train model** | `python train_comprehensive.py` |
| **Test model** | `python utils.py test` |
| **Generate graphs** | `python utils.py graphs` |

---

## 📡 API REFERENCE

### Base URL
```
https://localhost:5000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

#### 2. Predict Audio Authenticity
```http
POST /predict
Content-Type: multipart/form-data
```

**Parameters:**
- `file`: Audio file (WAV, FLAC, MP3, M4A, OGG)
- Max size: 100MB
- Recommended: 4-second clips at 16kHz

**Response (Real Audio):**
```json
{
  "is_fake": false,
  "confidence": 0.9234,
  "prediction": "REAL"
}
```

**Response (Fake Audio):**
```json
{
  "is_fake": true,
  "confidence": 0.8712,
  "prediction": "FAKE"
}
```

---

## 🏗️ SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                   USER INTERFACE (React)                    │
│            https://localhost:3000                           │
│         [Drag & Drop] [Upload] [Results Display]           │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTPS Request
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   FLASK REST API                            │
│              https://localhost:5000                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. Receive & Validate Audio                        │   │
│  │  2. Feature Extraction (LFCC, Spectral)             │   │
│  │  3. AASIST Model Inference                          │   │
│  │  4. Return Prediction + Confidence                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────┐
│   AASIST MODEL (GPU)    │
│   5.4M Parameters       │
│   CUDA Accelerated      │
└─────────────────────────┘
```

---

## 🛠️ DEVELOPMENT

### Hardware Requirements

**Minimum:**
- CPU: Multi-core processor (Intel i5 or equivalent)
- RAM: 8GB
- Storage: 10GB free space

**Recommended:**
- CPU: Intel i7/AMD Ryzen 7 or better
- RAM: 16GB
- GPU: NVIDIA GPU with 4GB+ VRAM (CUDA 11.8+)
- Storage: 20GB free space (for datasets)

### Dataset Information

**Datasets (not included in repository):**
- **ASVspoof2019 LA**: ~7.3 GB - Download from [official source](https://datashare.ed.ac.uk/handle/10283/3336)
- **LibriSpeech test-clean**: ~350 MB - Auto-download via `python utils.py download`

---

## 🔒 SECURITY

The application uses self-signed SSL certificates for HTTPS:
- **Location**: `certificates/cert.pem`, `certificates/key.pem`
- **Note**: Browsers will show security warnings (expected)
- **Production**: Replace with proper SSL certificates from Let's Encrypt or CA

---

## 📈 PERFORMANCE BENCHMARKS

| Hardware | Inference Time | Throughput |
|----------|---------------|------------|
| **NVIDIA RTX 2050** | 14.8ms | 67 clips/sec |
| **NVIDIA RTX 3060** | 9.2ms | 108 clips/sec |
| **Intel i7 CPU** | 156ms | 6.4 clips/sec |

*Based on 4-second audio clips at 16kHz*

---

## 🐛 TROUBLESHOOTING

**Issue: ModuleNotFoundError**
```bash
pip install -r backend/requirements.txt
```

**Issue: Port already in use**
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**Issue: SSL Certificate Warning**
- Expected for self-signed certificates
- Click "Advanced" → "Proceed to localhost"

---

## 🤝 CONTRIBUTING

Contributions welcome! Areas for improvement:
- Real-time microphone input
- Batch processing API
- Docker containerization
- Mobile app integration
- Additional TTS detection systems

---

## 📄 LICENSE

MIT License - See LICENSE file for details

---

## 🙏 ACKNOWLEDGMENTS

- **ASVspoof 2019 Challenge** - Comprehensive spoofing dataset
- **AASIST Authors** - State-of-the-art anti-spoofing architecture
- **LibriSpeech** - Diverse real speech samples
- **PyTorch Team** - Deep learning framework

---

## 📚 REFERENCES

1. Jung, Jee-weon, et al. "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks." *ICASSP 2022*.
2. ASVspoof 2019: "The ASVspoof 2019 database." *Zenodo*, 2019.
3. LibriSpeech: Panayotov, V., et al. "Librispeech: An ASR corpus based on public domain audio books." *ICASSP 2015*.

---

<div align="center">

**🔐 Built for AI Security Research**

**Repository**: [github.com/anjo3902/Voice_Deepfake_Detection](https://github.com/anjo3902/Voice_Deepfake_Detection)

⭐ Star this repo if you find it helpful!

</div>

