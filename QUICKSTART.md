# Quick Reference Guide - Dermato AI

## 📋 Command Cheat Sheet

### Setup & Installation
```bash
# Verify environment
python setup.py

# Install dependencies
pip install -r requirements_updated.txt

# Install MindSpore (CPU)
pip install mindspore==2.2.0
```

### Training
```bash
# Train ResNet with MindSpore
python train_mindspore.py --model resnet --epochs 30 --batch_size 32

# Train EfficientNet with MindSpore
python train_mindspore.py --model efficientnet --epochs 30

# Resume from checkpoint
python train_mindspore.py --resume checkpoints/skin_lesion_resnet-5_500.ckpt --epochs 10

# Train with TensorFlow (original)
python train.py
```

### Inference
```bash
# Auto-select framework
python inference.py --image path/to/image.jpg

# Use TensorFlow
python inference.py --image path/to/image.jpg --framework tensorflow

# Use MindSpore ResNet
python inference.py --image path/to/image.jpg --framework mindspore --model_type resnet

# Use Ensemble
python inference.py --image path/to/image.jpg --framework ensemble
```

### Evaluation
```bash
# Evaluate MindSpore model
python evaluate_improved.py --framework mindspore --model_type resnet --checkpoint checkpoints/best.ckpt

# Evaluate TensorFlow model
python evaluate_improved.py --framework tensorflow
```

### API Server
```bash
# Start improved server (recommended)
python main_improved.py

# Start original server
python main.py
```

### Visualization
```bash
# Plot class distribution
python utils.py --action class_dist --output plots/distribution.png

# Plot sample images
python utils.py --action samples --output plots/samples.png

# Plot training history
python utils.py --action history --output plots/history.png
```

## 🔌 API Endpoints

### Health Check
```bash
curl http://localhost:5000/health
```

### List Models
```bash
curl http://localhost:5000/models
```

### Single Prediction
```bash
# Basic
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict

# With options
curl -X POST \
  -F "file=@image.jpg" \
  -F "framework=mindspore" \
  -F "model_type=resnet" \
  -F "get_advice=true" \
  http://localhost:5000/predict
```

### Batch Prediction
```bash
curl -X POST \
  -F "files[]=@img1.jpg" \
  -F "files[]=@img2.jpg" \
  -F "framework=mindspore" \
  http://localhost:5000/predict_batch
```

## 🐍 Python API Examples

### Simple Prediction
```python
from inference import predict_single_image

result = predict_single_image('image.jpg', framework='auto')
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Using Specific Framework
```python
from inference import TensorFlowInference, MindSporeInference

# TensorFlow
tf_model = TensorFlowInference()
result = tf_model.predict('image.jpg')

# MindSpore
ms_model = MindSporeInference(model_type='resnet')
result = ms_model.predict('image.jpg')
```

### Ensemble Prediction
```python
from inference import EnsembleInference

ensemble = EnsembleInference()
result = ensemble.predict('image.jpg', weights=(0.6, 0.4))
```

### Batch Processing
```python
from inference import predict_single_image
import os

images_dir = 'test_images/'
results = []

for filename in os.listdir(images_dir):
    if filename.endswith(('.jpg', '.png')):
        path = os.path.join(images_dir, filename)
        result = predict_single_image(path, framework='mindspore')
        results.append({
            'filename': filename,
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence']
        })

for r in results:
    print(f"{r['filename']}: {r['predicted_class']} ({r['confidence']:.2%})")
```

## ⚙️ Configuration

### Edit config.py
```python
# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30

# Model paths
TF_MODEL_PATH = 'model.h5'
MINDSPORE_MODEL_PATH = 'model_mindspore.ckpt'

# Device settings
MINDSPORE_DEVICE_TARGET = "CPU"  # or "GPU", "Ascend"
```

## 📊 Model Classes

| ID | Label | Description |
|----|-------|-------------|
| 0 | akiec | Actinic Keratoses |
| 1 | bcc | Basal Cell Carcinoma |
| 2 | bkl | Benign Keratosis |
| 3 | df | Dermatofibroma |
| 4 | mel | Melanoma |
| 5 | nv | Melanocytic Nevi |
| 6 | vasc | Vascular lesion |

## 🗂️ File Structure
```
Dermato_Ai/
├── config.py                    # ⚙️ Configuration
├── model_mindspore.py           # 🧠 MindSpore models
├── train_mindspore.py           # 🏋️ Training script
├── data_preprocessing_mindspore.py  # 📊 Data pipeline
├── inference.py                 # 🔮 Inference engine
├── main_improved.py             # 🌐 API server
├── evaluate_improved.py         # 📈 Evaluation
├── utils.py                     # 🔧 Utilities
├── setup.py                     # 🚀 Setup script
├── README.md                    # 📖 Documentation
├── IMPROVEMENTS.md              # 📝 Changes log
├── requirements_updated.txt     # 📦 Dependencies
├── checkpoints/                 # 💾 Saved models
├── logs/                        # 📄 Training logs
└── uploads/                     # 📁 Uploaded files
```

## 🔍 Troubleshooting

### Import Errors
```bash
# Install missing packages
pip install <package_name>

# Verify installation
python -c "import mindspore; print(mindspore.__version__)"
```

### CUDA Errors (GPU)
```bash
# Check CUDA version
nvidia-smi

# Install correct MindSpore GPU version
# For CUDA 11.6:
pip install mindspore-gpu==2.2.0
```

### Dataset Not Found
```bash
# Download HAM10000 from Kaggle
# Place in:
# - HAM10000_images/
# - HAM10000_metadata.csv
```

### Model Loading Errors
```bash
# Check model file exists
ls -lh model.h5
ls -lh model_mindspore.ckpt

# Verify paths in config.py
```

### API Connection Errors
```bash
# Check if server is running
curl http://localhost:5000/health

# Start server
python main_improved.py
```

## 💡 Tips & Best Practices

### Training
- Start with smaller epochs (10-15) to test
- Use early stopping to prevent overfitting
- Monitor validation accuracy
- Save checkpoints frequently
- Use GPU/Ascend for faster training

### Inference
- Use ensemble for best accuracy
- Cache models to avoid reloading
- Batch process multiple images
- Set appropriate confidence threshold

### Deployment
- Use main_improved.py for production
- Enable logging for monitoring
- Set up error alerts
- Use reverse proxy (nginx) for scale
- Implement rate limiting

### Performance
- Use GRAPH mode for production
- Enable mixed precision (FP16)
- Optimize batch size for hardware
- Use dataset sink mode
- Profile bottlenecks

## 📚 Resources

- [MindSpore Docs](https://www.mindspore.cn/en)
- [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [CBAM Paper](https://arxiv.org/abs/1807.06521)
- [Focal Loss Paper](https://arxiv.org/abs/1708.02002)

## ❓ FAQ

**Q: Which framework should I use?**
A: MindSpore for new projects (better features), TensorFlow for compatibility, Ensemble for best accuracy.

**Q: How much data do I need?**
A: HAM10000 has ~10,000 images. More data = better performance.

**Q: Can I add more classes?**
A: Yes, update NUM_CLASSES in config.py and retrain.

**Q: How to improve accuracy?**
A: More training data, longer training, ensemble models, tune hyperparameters.

**Q: Production deployment?**
A: Use main_improved.py + nginx + gunicorn for scalability.

---

**Need help? Check README.md or open an issue!**
