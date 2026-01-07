"""
Configuration file for Dermato AI - Skin Lesion Classification
"""

import os

# Dataset Configuration
DATA_DIR = 'HAM10000_images'
METADATA_FILE = 'HAM10000_metadata.csv'
UPLOAD_FOLDER = 'uploads'

# Model Configuration
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 7
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30

# Class names for skin lesions
CLASS_NAMES = {
    0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen's disease)',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular skin lesion'
}

# Label mapping (corresponds to HAM10000 dataset)
LABEL_MAPPING = {
    'akiec': 0,  # Actinic Keratoses
    'bcc': 1,    # Basal Cell Carcinoma
    'bkl': 2,    # Benign Keratosis
    'df': 3,     # Dermatofibroma
    'mel': 4,    # Melanoma
    'nv': 5,     # Melanocytic Nevi
    'vasc': 6    # Vascular lesion
}

# Training Configuration
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Model paths
TF_MODEL_PATH = 'model.h5'
MINDSPORE_MODEL_PATH = 'model_mindspore.ckpt'
BEST_MODEL_PATH = 'best_model.ckpt'

# Prediction Configuration
CONFIDENCE_THRESHOLD = 0.8

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.15,
    'height_shift_range': 0.15,
    'shear_range': 0.15,
    'zoom_range': 0.15,
    'horizontal_flip': True,
    'vertical_flip': True,
    'brightness_range': (0.8, 1.2)
}

# MindSpore Configuration
MINDSPORE_DEVICE_TARGET = "CPU"  # Change to "GPU" or "Ascend" if available
MINDSPORE_DEVICE_ID = 0

# Training callbacks configuration
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# Logging
LOG_DIR = 'logs'
CHECKPOINT_DIR = 'checkpoints'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
