"""
Utility functions for visualization and analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import config


def plot_training_history(history_file, save_path=None):
    """
    Plot training history from log file
    
    Args:
        history_file: Path to training log CSV file
        save_path: Where to save the plot
    """
    import pandas as pd
    
    # Load history
    df = pd.read_csv(history_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axes[0, 0].plot(df['Epoch'], df['Train_Loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(df['Epoch'], df['Val_Loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot accuracy
    axes[0, 1].plot(df['Epoch'], df['Train_Acc'], label='Train Acc', marker='o')
    axes[0, 1].plot(df['Epoch'], df['Val_Acc'], label='Val Acc', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot time per epoch
    axes[1, 0].bar(df['Epoch'], df['Time'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Training Time per Epoch')
    axes[1, 0].grid(alpha=0.3)
    
    # Plot learning curve
    axes[1, 1].plot(df['Epoch'], df['Val_Acc'], marker='o', linewidth=2)
    axes[1, 1].fill_between(df['Epoch'], 
                             df['Val_Acc'] - df['Val_Acc'].std(), 
                             df['Val_Acc'] + df['Val_Acc'].std(),
                             alpha=0.3)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].set_title('Learning Curve')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()


def visualize_predictions(images, true_labels, pred_labels, probabilities, 
                         num_samples=16, save_path=None):
    """
    Visualize predictions with images
    
    Args:
        images: Array of images
        true_labels: True labels
        pred_labels: Predicted labels
        probabilities: Prediction probabilities
        num_samples: Number of samples to visualize
        save_path: Where to save the plot
    """
    num_samples = min(num_samples, len(images))
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Display image
        img = images[i]
        if img.shape[0] == 3:  # CHW format
            img = np.transpose(img, (1, 2, 0))
        
        # Denormalize if needed
        if img.max() <= 1.0:
            img = img * 255
        
        ax.imshow(img.astype(np.uint8))
        
        # Add title with prediction info
        true_class = config.CLASS_NAMES[true_labels[i]]
        pred_class = config.CLASS_NAMES[pred_labels[i]]
        confidence = probabilities[i][pred_labels[i]]
        
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        title = f"True: Class {true_labels[i]}\n"
        title += f"Pred: Class {pred_labels[i]} ({confidence:.2%})"
        
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to {save_path}")
    else:
        plt.show()


def plot_class_distribution(metadata_file, save_path=None):
    """
    Plot distribution of classes in dataset
    
    Args:
        metadata_file: Path to metadata CSV
        save_path: Where to save the plot
    """
    import pandas as pd
    
    df = pd.read_csv(metadata_file)
    
    plt.figure(figsize=(12, 6))
    
    # Count distribution
    class_counts = df['dx'].value_counts()
    
    # Create bar plot
    ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
    
    plt.xlabel('Disease Type', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Distribution of Skin Lesion Classes', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 50, str(v), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    else:
        plt.show()


def plot_sample_images(data_dir, metadata_file, num_per_class=3, save_path=None):
    """
    Plot sample images from each class
    
    Args:
        data_dir: Directory with images
        metadata_file: Metadata CSV file
        num_per_class: Number of samples per class
        save_path: Where to save the plot
    """
    import pandas as pd
    
    df = pd.read_csv(metadata_file)
    df['image_id'] = df['image_id'].apply(lambda x: f"{x}.jpg" if not x.endswith('.jpg') else x)
    
    classes = df['dx'].unique()
    
    fig, axes = plt.subplots(len(classes), num_per_class, 
                             figsize=(num_per_class * 4, len(classes) * 3))
    
    for i, cls in enumerate(classes):
        class_df = df[df['dx'] == cls].sample(n=num_per_class)
        
        for j, (_, row) in enumerate(class_df.iterrows()):
            ax = axes[i, j] if len(classes) > 1 else axes[j]
            
            # Load and display image
            img_path = os.path.join(data_dir, row['image_id'])
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                
                if j == 0:
                    ax.set_ylabel(cls, fontsize=12, rotation=0, 
                                 labelpad=40, va='center')
                
                ax.set_title(f"ID: {row['image_id'][:8]}...", fontsize=10)
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {str(e)}", 
                       ha='center', va='center')
                ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample images plot saved to {save_path}")
    else:
        plt.show()


def compare_model_predictions(image_path, tf_model=None, ms_model=None):
    """
    Compare predictions from different models on the same image
    
    Args:
        image_path: Path to image
        tf_model: TensorFlow model instance
        ms_model: MindSpore model instance
    """
    from inference import TensorFlowInference, MindSporeInference
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display image
    img = Image.open(image_path)
    axes[0].imshow(img)
    axes[0].set_title('Input Image', fontsize=14)
    axes[0].axis('off')
    
    # TensorFlow predictions
    if tf_model is not None:
        tf_result = TensorFlowInference().predict(image_path)
        probs = [tf_result['all_probabilities'][config.CLASS_NAMES[i]] 
                for i in range(config.NUM_CLASSES)]
        
        axes[1].barh(range(config.NUM_CLASSES), probs, color='skyblue')
        axes[1].set_yticks(range(config.NUM_CLASSES))
        axes[1].set_yticklabels([f"Class {i}" for i in range(config.NUM_CLASSES)])
        axes[1].set_xlabel('Probability', fontsize=12)
        axes[1].set_title(f'TensorFlow\nPredicted: Class {tf_result["predicted_class"]}\n'
                         f'Confidence: {tf_result["confidence"]:.2%}', fontsize=12)
        axes[1].set_xlim(0, 1)
    
    # MindSpore predictions
    if ms_model is not None:
        ms_result = MindSporeInference().predict(image_path)
        probs = [ms_result['all_probabilities'][config.CLASS_NAMES[i]] 
                for i in range(config.NUM_CLASSES)]
        
        axes[2].barh(range(config.NUM_CLASSES), probs, color='lightcoral')
        axes[2].set_yticks(range(config.NUM_CLASSES))
        axes[2].set_yticklabels([f"Class {i}" for i in range(config.NUM_CLASSES)])
        axes[2].set_xlabel('Probability', fontsize=12)
        axes[2].set_title(f'MindSpore\nPredicted: Class {ms_result["predicted_class"]}\n'
                         f'Confidence: {ms_result["confidence"]:.2%}', fontsize=12)
        axes[2].set_xlim(0, 1)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualization utilities')
    parser.add_argument('--action', type=str, required=True,
                       choices=['class_dist', 'samples', 'history'],
                       help='What to visualize')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    
    args = parser.parse_args()
    
    if args.action == 'class_dist':
        plot_class_distribution(config.METADATA_FILE, save_path=args.output)
    
    elif args.action == 'samples':
        plot_sample_images(config.DATA_DIR, config.METADATA_FILE, 
                          num_per_class=3, save_path=args.output)
    
    elif args.action == 'history':
        log_file = os.path.join(config.LOG_DIR, 'training_resnet.log')
        if os.path.exists(log_file):
            plot_training_history(log_file, save_path=args.output)
        else:
            print(f"Log file not found: {log_file}")
