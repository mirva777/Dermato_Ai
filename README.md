# Dermato AI - Skin Lesion Classification 🔬

An advanced AI-powered skin lesion classification system using both **TensorFlow** and **Huawei's MindSpore** frameworks. This project implements state-of-the-art deep learning models for automated detection and classification of seven types of skin lesions.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![MindSpore](https://img.shields.io/badge/MindSpore-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

### Model Architectures
- **TensorFlow/Keras**: EfficientNetB0 with transfer learning
- **MindSpore**: Custom ResNet with CBAM attention mechanisms
- **MindSpore**: Lightweight EfficientNet implementation
- **Ensemble**: Combined predictions from both frameworks

### Advanced Training Features
- ✅ Learning rate scheduling with warmup and cosine annealing
- ✅ Focal loss for handling class imbalance
- ✅ Label smoothing for better generalization
- ✅ Channel and spatial attention (CBAM)
- ✅ Early stopping and model checkpointing
- ✅ Comprehensive metrics tracking
- ✅ Data augmentation (rotation, flip, color adjustment)

### Web API
- ✅ Flask-based REST API
- ✅ Single and batch image prediction
- ✅ Health check and model listing endpoints
- ✅ Medical advice integration
- ✅ Comprehensive error handling and logging
- ✅ CORS support for frontend integration

## 📋 Skin Lesion Classes

The model classifies images into 7 categories:

1. **Actinic Keratoses (akiec)** - Solar Keratoses or intraepithelial Carcinoma
2. **Basal Cell Carcinoma (bcc)** - Most common skin cancer
3. **Benign Keratosis (bkl)** - Non-cancerous skin growth
4. **Dermatofibroma (df)** - Benign fibrous nodule
5. **Melanoma (mel)** - Serious skin cancer
6. **Melanocytic Nevi (nv)** - Common moles
7. **Vascular skin lesion (vasc)** - Blood vessel abnormalities

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- CUDA 11.6+ (for GPU support)
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Dermato_Ai
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**

For **TensorFlow only**:
```bash
pip install -r requirements.txt
```

For **TensorFlow + MindSpore**:
```bash
pip install -r requirements_updated.txt
```

**Note**: MindSpore installation may vary by platform. See [MindSpore Installation Guide](https://www.mindspore.cn/install/en).

4. **Download dataset**

Download the HAM10000 dataset:
- [Kaggle HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Place images in `HAM10000_images/`
- Place `HAM10000_metadata.csv` in the root directory

## 🏋️ Training

### Using MindSpore (Recommended)

**Train ResNet with CBAM attention**:
```bash
python train_mindspore.py --model resnet --epochs 30 --batch_size 32
```

**Train EfficientNet**:
```bash
python train_mindspore.py --model efficientnet --epochs 30
```

**Resume training from checkpoint**:
```bash
python train_mindspore.py --resume checkpoints/skin_lesion_resnet-5_500.ckpt --epochs 10
```

### Using TensorFlow (Legacy)

```bash
python train.py
```

### Training Features

The MindSpore training script includes:
- **Learning Rate Scheduler**: Warmup + cosine annealing
- **Mixed Loss**: Focal loss (70%) + Label smoothing (30%)
- **Class Weights**: Automatic balancing for imbalanced dataset
- **Callbacks**: Early stopping, model checkpointing, metrics logging
- **Metrics**: Accuracy, Precision, Recall tracking

Training logs are saved to `logs/training_{model_type}.log`.

## 🔮 Inference

### Command Line

**Using TensorFlow**:
```bash
python inference.py --image path/to/image.jpg --framework tensorflow
```

**Using MindSpore**:
```bash
python inference.py --image path/to/image.jpg --framework mindspore --model_type resnet
```

**Using Ensemble (both frameworks)**:
```bash
python inference.py --image path/to/image.jpg --framework ensemble
```

### Python API

```python
from inference import predict_single_image

# Predict with auto-selection
result = predict_single_image('image.jpg', framework='auto')

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")

# Print top 3 predictions
for pred in result['top_predictions']:
    print(f"{pred['class']}: {pred['probability']:.2%}")
```

### Using Specific Inference Classes

```python
from inference import TensorFlowInference, MindSporeInference

# TensorFlow
tf_model = TensorFlowInference()
result = tf_model.predict('image.jpg')

# MindSpore
ms_model = MindSporeInference(model_type='resnet')
result = ms_model.predict('image.jpg')
```

## 🌐 Web API

### Start the Server

**Improved version (recommended)**:
```bash
python main_improved.py
```

**Original version**:
```bash
python main.py
```

The API will be available at `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

#### 2. List Available Models
```bash
GET /models
```

#### 3. Single Prediction
```bash
POST /predict

Form Data:
- file: Image file
- framework: 'tensorflow', 'mindspore', or 'auto' (optional)
- model_type: 'resnet' or 'efficientnet' (optional)
- get_advice: 'true' or 'false' (optional)
```

**Example using curl**:
```bash
curl -X POST -F "file=@image.jpg" -F "framework=mindspore" http://localhost:5000/predict
```

**Example using Python requests**:
```python
import requests

url = 'http://localhost:5000/predict'
files = {'file': open('image.jpg', 'rb')}
data = {'framework': 'mindspore', 'model_type': 'resnet'}

response = requests.post(url, files=files, data=data)
print(response.json())
```

#### 4. Batch Prediction
```bash
POST /predict_batch

Form Data:
- files[]: Multiple image files
- framework: Model framework (optional)
```

### API Response Format

```json
{
  "status": "success",
  "predicted_class": "Melanoma",
  "confidence": 0.92,
  "framework": "MindSpore",
  "top_predictions": [
    {"class": "Melanoma", "probability": 0.92},
    {"class": "Melanocytic Nevi", "probability": 0.05},
    {"class": "Basal Cell Carcinoma", "probability": 0.02}
  ],
  "medical_advice": "...",
  "metadata": {
    "filename": "20260107_120000_image.jpg",
    "timestamp": "2026-01-07T12:00:00",
    "processing_time": 0.234
  }
}
```

## 📊 Model Architecture

### MindSpore ResNet-CBAM

```
Input (224x224x3)
    ↓
Initial Conv (7x7, stride=2)
    ↓
MaxPool (3x3, stride=2)
    ↓
Residual Layer 1 (64 channels, 2 blocks)
    ↓
Residual Layer 2 (128 channels, 2 blocks)
    ↓
Residual Layer 3 (256 channels, 2 blocks)
    ↓
Residual Layer 4 (512 channels, 2 blocks)
    ↓ (each block has CBAM attention)
Global Average Pooling
    ↓
FC (512 → 256) + BN + ReLU + Dropout
    ↓
FC (256 → 128) + BN + ReLU + Dropout
    ↓
FC (128 → 7) + Softmax
```

**Features**:
- Residual connections for better gradient flow
- CBAM (Channel + Spatial Attention)
- Batch normalization for stable training
- Dropout for regularization
- He initialization

### TensorFlow EfficientNetB0

Pre-trained EfficientNetB0 with custom classification head.

## 📈 Performance Optimization

### MindSpore Advantages
1. **Memory Efficiency**: Optimized memory management
2. **Hardware Support**: Native support for Ascend NPU
3. **Dynamic Graph**: Better debugging with PYNATIVE mode
4. **Static Graph**: Better performance with GRAPH mode

### Training Tips
- Use **mixed precision** for faster training (FP16)
- Enable **dataset sink mode** for faster data loading
- Use **distributed training** for multi-GPU setups
- Tune **batch size** based on available memory

## 🔧 Configuration

All configuration is centralized in [config.py](config.py):

```python
# Dataset
DATA_DIR = 'HAM10000_images'
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 7

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30

# MindSpore
MINDSPORE_DEVICE_TARGET = "CPU"  # or "GPU", "Ascend"

# Paths
TF_MODEL_PATH = 'model.h5'
MINDSPORE_MODEL_PATH = 'model_mindspore.ckpt'
```

## 📁 Project Structure

```
Dermato_Ai/
├── config.py                          # Configuration settings
├── model_mindspore.py                 # MindSpore model architectures
├── model.py                           # TensorFlow model (legacy)
├── train_mindspore.py                 # MindSpore training script
├── train.py                           # TensorFlow training (legacy)
├── data_preprocessing_mindspore.py    # MindSpore data pipeline
├── data_preprocessing.py              # TensorFlow data pipeline
├── inference.py                       # Unified inference module
├── main_improved.py                   # Improved Flask API
├── main.py                            # Original Flask API
├── requirements_updated.txt           # Dependencies with MindSpore
├── requirements.txt                   # Original dependencies
├── checkpoints/                       # Saved model checkpoints
├── logs/                              # Training and API logs
├── uploads/                           # Uploaded images
└── templates/
    └── index.html                     # Web interface
```

## 🧪 Testing

### Test Data Preprocessing
```bash
python data_preprocessing_mindspore.py
```

### Test Model Creation
```bash
python model_mindspore.py
```

### Test Inference
```bash
python inference.py --image test_image.jpg
```

## 🎯 Use Cases

1. **Telemedicine**: Remote skin screening
2. **Clinical Decision Support**: Assist dermatologists
3. **Public Health**: Mass screening programs
4. **Medical Education**: Training tool for students
5. **Self-Assessment**: Consumer health applications

## ⚠️ Important Disclaimers

- This AI tool is for **screening purposes only**
- **NOT a replacement** for professional medical diagnosis
- Always consult a qualified dermatologist
- AI predictions may have false positives/negatives
- No liability accepted for medical decisions

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **HAM10000 Dataset**: Tschandl, P., Rosendahl, C. & Kittler, H.
- **MindSpore**: Huawei Technologies Co., Ltd.
- **TensorFlow**: Google Brain Team
- **EfficientNet**: Tan, M. & Le, Q. V.

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🔗 Resources

- [MindSpore Documentation](https://www.mindspore.cn/en)
- [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
- [HAM10000 Dataset Paper](https://arxiv.org/abs/1803.10417)
- [CBAM Attention Paper](https://arxiv.org/abs/1807.06521)

---

**Made with ❤️ for better healthcare through AI**
