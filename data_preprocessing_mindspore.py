"""
MindSpore Data Preprocessing Pipeline for Skin Lesion Classification
Optimized data loading, augmentation, and batching
"""

import os
import numpy as np
import pandas as pd
import mindspore
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.dataset.vision import Inter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import config


class SkinLesionDataset:
    """Custom dataset for skin lesion images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(config.DATA_DIR, self.image_paths[idx])
        
        # Read image as numpy array
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        label = self.labels[idx]
        
        return image, label


def load_and_preprocess_metadata():
    """
    Load metadata and prepare for training
    
    Returns:
        train_df, val_df, test_df, class_weights
    """
    # Load metadata
    metadata = pd.read_csv(config.METADATA_FILE)
    
    # Add image extension
    metadata['image_id'] = metadata['image_id'].apply(lambda x: f"{x}.jpg")
    
    # Map labels to integers
    metadata['label'] = metadata['dx'].map(config.LABEL_MAPPING)
    
    # Check for missing values
    print(f"Total images: {len(metadata)}")
    print(f"Missing values:\n{metadata.isnull().sum()}")
    print(f"\nClass distribution:\n{metadata['dx'].value_counts()}")
    
    # Split data
    train_val_df, test_df = train_test_split(
        metadata, 
        test_size=config.TEST_SPLIT, 
        stratify=metadata['label'],
        random_state=42
    )
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=config.VAL_SPLIT / (1 - config.TEST_SPLIT),
        stratify=train_val_df['label'],
        random_state=42
    )
    
    print(f"\nData split:")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Compute class weights for handling imbalanced dataset
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label']
    )
    
    print(f"\nClass weights: {class_weights}")
    
    return train_df, val_df, test_df, class_weights


def create_transforms(is_training=True):
    """
    Create data augmentation and preprocessing transforms
    
    Args:
        is_training: Whether to apply training augmentations
        
    Returns:
        List of transforms
    """
    if is_training:
        transform_list = [
            vision.Decode(),
            vision.Resize(config.IMAGE_SIZE),
            vision.RandomRotation(degrees=config.AUGMENTATION_CONFIG['rotation_range']),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.RandomVerticalFlip(prob=0.5),
            vision.RandomColorAdjust(
                brightness=(0.8, 1.2),
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2)
            ),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vision.HWC2CHW()
        ]
    else:
        transform_list = [
            vision.Decode(),
            vision.Resize(config.IMAGE_SIZE),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vision.HWC2CHW()
        ]
    
    return transform_list


def create_dataset(df, is_training=True, batch_size=None, num_parallel_workers=4):
    """
    Create MindSpore dataset from DataFrame
    
    Args:
        df: DataFrame with image_id and label columns
        is_training: Whether this is a training dataset
        batch_size: Batch size (uses config default if None)
        num_parallel_workers: Number of parallel workers for data loading
        
    Returns:
        MindSpore dataset
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # Prepare data
    image_paths = df['image_id'].values.tolist()
    labels = df['label'].values.astype(np.int32).tolist()
    
    # Create generator dataset
    dataset = ds.GeneratorDataset(
        source=[(os.path.join(config.DATA_DIR, img), label) 
                for img, label in zip(image_paths, labels)],
        column_names=["image", "label"],
        shuffle=is_training,
        num_parallel_workers=num_parallel_workers
    )
    
    # Apply transforms
    transform_list = create_transforms(is_training)
    
    # Image transforms
    for transform in transform_list[:-1]:  # All except HWC2CHW
        dataset = dataset.map(
            operations=transform,
            input_columns=["image"],
            num_parallel_workers=num_parallel_workers
        )
    
    # Final transform
    dataset = dataset.map(
        operations=transform_list[-1],  # HWC2CHW
        input_columns=["image"],
        num_parallel_workers=num_parallel_workers
    )
    
    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    
    return dataset


def create_simple_dataset(image_dir, label_file, is_training=True):
    """
    Simplified dataset creation using built-in MindSpore functions
    
    Args:
        image_dir: Directory containing images
        label_file: CSV file with labels
        is_training: Whether this is a training dataset
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Load metadata
    train_df, val_df, test_df, class_weights = load_and_preprocess_metadata()
    
    # Create datasets
    train_dataset = create_dataset(train_df, is_training=True)
    val_dataset = create_dataset(val_df, is_training=False)
    test_dataset = create_dataset(test_df, is_training=False)
    
    return train_dataset, val_dataset, test_dataset, class_weights


class AdvancedAugmentation:
    """Advanced augmentation techniques for medical images"""
    
    @staticmethod
    def mixup(images, labels, alpha=0.2):
        """
        MixUp augmentation
        
        Args:
            images: Batch of images
            labels: Batch of labels
            alpha: MixUp parameter
            
        Returns:
            Mixed images and labels
        """
        batch_size = images.shape[0]
        lam = np.random.beta(alpha, alpha)
        
        index = np.random.permutation(batch_size)
        mixed_images = lam * images + (1 - lam) * images[index]
        
        return mixed_images, labels, labels[index], lam
    
    @staticmethod
    def cutmix(images, labels, alpha=1.0):
        """
        CutMix augmentation
        
        Args:
            images: Batch of images
            labels: Batch of labels
            alpha: CutMix parameter
            
        Returns:
            Mixed images and labels
        """
        batch_size = images.shape[0]
        lam = np.random.beta(alpha, alpha)
        
        index = np.random.permutation(batch_size)
        
        # Generate random box
        _, _, h, w = images.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix
        mixed_images = images.copy()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        return mixed_images, labels, labels[index], lam


def preprocess_single_image(image_path, target_size=config.IMAGE_SIZE):
    """
    Preprocess a single image for inference
    
    Args:
        image_path: Path to image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image tensor
    """
    from PIL import Image
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image = np.array(image, dtype=np.float32)
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255
    image = (image - mean) / std
    
    # Convert to CHW format
    image = np.transpose(image, (2, 0, 1))
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return mindspore.Tensor(image, mindspore.float32)


# Example usage
if __name__ == "__main__":
    print("Creating datasets...")
    
    try:
        train_dataset, val_dataset, test_dataset, class_weights = create_simple_dataset(
            config.DATA_DIR,
            config.METADATA_FILE,
            is_training=True
        )
        
        print(f"\nDataset creation successful!")
        print(f"Train batches: {train_dataset.get_dataset_size()}")
        print(f"Validation batches: {val_dataset.get_dataset_size()}")
        print(f"Test batches: {test_dataset.get_dataset_size()}")
        
        # Test one batch
        for data in train_dataset.create_dict_iterator():
            images = data['image']
            labels = data['label']
            print(f"\nBatch shape - Images: {images.shape}, Labels: {labels.shape}")
            print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
            break
            
    except Exception as e:
        print(f"Error creating datasets: {e}")
        print("Make sure the dataset directory and metadata file exist.")
