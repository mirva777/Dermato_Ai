"""
Evaluation script for trained models
Computes metrics on test set and generates visualizations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
import config


def evaluate_model(model, test_dataset, framework='mindspore', save_plots=True):
    """
    Evaluate model on test dataset
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        framework: 'mindspore' or 'tensorflow'
        save_plots: Whether to save visualization plots
        
    Returns:
        Dictionary with evaluation metrics
    """
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    if framework == 'mindspore':
        import mindspore.nn as nn
        from mindspore import Tensor
        
        softmax = nn.Softmax(axis=1)
        model.set_train(False)
        
        for data in test_dataset.create_dict_iterator():
            images = data['image']
            labels = data['label'].asnumpy()
            
            # Get predictions
            logits = model(images)
            probs = softmax(logits).asnumpy()
            preds = np.argmax(probs, axis=1)
            
            all_predictions.extend(preds)
            all_labels.extend(labels)
            all_probabilities.extend(probs)
            
    elif framework == 'tensorflow':
        for images, labels in test_dataset:
            probs = model.predict(images, verbose=0)
            preds = np.argmax(probs, axis=1)
            
            all_predictions.extend(preds)
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probs)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Classification report
    class_names_list = [config.CLASS_NAMES[i] for i in range(config.NUM_CLASSES)]
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_predictions,
        target_names=class_names_list,
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    if save_plots:
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[f"Class {i}" for i in range(config.NUM_CLASSES)],
            yticklabels=[f"Class {i}" for i in range(config.NUM_CLASSES)]
        )
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(config.LOG_DIR, f'confusion_matrix_{framework}.png'), dpi=300)
        print(f"\nConfusion matrix saved to logs/confusion_matrix_{framework}.png")
        
        # Plot per-class accuracy
        plt.figure(figsize=(14, 6))
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        plt.bar(range(config.NUM_CLASSES), class_accuracy, color='skyblue')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Per-Class Accuracy', fontsize=16)
        plt.xticks(range(config.NUM_CLASSES), [f"Class {i}" for i in range(config.NUM_CLASSES)], rotation=45)
        plt.ylim(0, 1)
        for i, v in enumerate(class_accuracy):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(config.LOG_DIR, f'class_accuracy_{framework}.png'), dpi=300)
        print(f"Per-class accuracy plot saved to logs/class_accuracy_{framework}.png")
        
        # ROC curves (one-vs-rest)
        plt.figure(figsize=(10, 8))
        
        for i in range(config.NUM_CLASSES):
            # Binarize labels for class i
            binary_labels = (all_labels == i).astype(int)
            
            # Get probabilities for class i
            class_probs = all_probabilities[:, i]
            
            # Calculate ROC
            fpr, tpr, _ = roc_curve(binary_labels, class_probs)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves (One-vs-Rest)', fontsize=16)
        plt.legend(loc='lower right', fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.LOG_DIR, f'roc_curves_{framework}.png'), dpi=300)
        print(f"ROC curves saved to logs/roc_curves_{framework}.png")
    
    # Calculate overall metrics
    accuracy = np.mean(all_predictions == all_labels)
    
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }
    
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--framework', type=str, default='mindspore',
                       choices=['mindspore', 'tensorflow'],
                       help='Framework to evaluate')
    parser.add_argument('--model_type', type=str, default='resnet',
                       choices=['resnet', 'efficientnet'],
                       help='Model type for MindSpore')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file')
    
    args = parser.parse_args()
    
    if args.framework == 'mindspore':
        import mindspore
        from mindspore import context, load_checkpoint, load_param_into_net
        from model_mindspore import create_model
        from data_preprocessing_mindspore import create_simple_dataset
        
        # Set context
        context.set_context(
            mode=context.GRAPH_MODE,
            device_target=config.MINDSPORE_DEVICE_TARGET
        )
        
        # Load dataset
        print("Loading test dataset...")
        _, _, test_dataset, _ = create_simple_dataset(
            config.DATA_DIR,
            config.METADATA_FILE
        )
        
        # Create model
        print(f"Creating {args.model_type} model...")
        model = create_model(model_type=args.model_type)
        
        # Load checkpoint
        if args.checkpoint:
            param_dict = load_checkpoint(args.checkpoint)
            load_param_into_net(model, param_dict)
            print(f"Loaded checkpoint: {args.checkpoint}")
        else:
            print("Warning: No checkpoint provided, using untrained model!")
        
        # Evaluate
        results = evaluate_model(model, test_dataset, framework='mindspore')
        
    elif args.framework == 'tensorflow':
        from tensorflow import keras
        from data_preprocessing import test_generator
        
        # Load model
        print("Loading TensorFlow model...")
        model = keras.models.load_model(config.TF_MODEL_PATH)
        
        # Evaluate
        results = evaluate_model(model, test_generator, framework='tensorflow')
    
    print("\nEvaluation complete!")
