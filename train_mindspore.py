"""
MindSpore Training Script for Skin Lesion Classification
Features: Learning rate scheduling, metrics tracking, checkpointing, and early stopping
"""

import os
import time
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, context, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.train.callback import Callback, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy, Precision, Recall
import config
from model_mindspore import create_model
from data_preprocessing_mindspore import create_simple_dataset


class MetricsCallback(Callback):
    """Custom callback for tracking and logging metrics"""
    
    def __init__(self, eval_dataset, model, log_file='training_log.txt'):
        super(MetricsCallback, self).__init__()
        self.eval_dataset = eval_dataset
        self.model = model
        self.log_file = os.path.join(config.LOG_DIR, log_file)
        self.best_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Create log file
        with open(self.log_file, 'w') as f:
            f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,Time\n")
    
    def epoch_end(self, run_context):
        """Called at the end of each epoch"""
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        
        # Get training metrics
        train_loss = cb_params.net_outputs
        
        # Evaluate on validation set
        start_time = time.time()
        metrics = self.model.eval(self.eval_dataset)
        eval_time = time.time() - start_time
        
        val_acc = metrics.get('Accuracy', 0.0)
        
        # Log metrics
        log_msg = f"Epoch [{epoch}/{cb_params.epoch_num}] - "
        log_msg += f"Train Loss: {train_loss:.4f}, "
        log_msg += f"Val Acc: {val_acc:.4f}, "
        log_msg += f"Time: {eval_time:.2f}s"
        
        print(log_msg)
        
        # Save to file
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},0.0,0.0,{val_acc:.4f},{eval_time:.2f}\n")
        
        # Save history
        self.history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            print(f"New best validation accuracy: {val_acc:.4f}")


class EarlyStoppingCallback(Callback):
    """Early stopping callback to stop training when validation loss stops improving"""
    
    def __init__(self, patience=7, min_delta=0.001):
        super(EarlyStoppingCallback, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def epoch_end(self, run_context):
        """Check if training should stop"""
        cb_params = run_context.original_args()
        current_loss = cb_params.net_outputs
        
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                self.early_stop = True
                run_context.request_stop()
        else:
            self.best_loss = current_loss
            self.counter = 0


class LearningRateScheduler(nn.learning_rate_schedule.LearningRateSchedule):
    """Custom learning rate scheduler with warmup and cosine annealing"""
    
    def __init__(self, base_lr, total_steps, warmup_steps=1000, min_lr=1e-7):
        super(LearningRateScheduler, self).__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
    
    def construct(self, global_step):
        """Calculate learning rate for current step"""
        if global_step < self.warmup_steps:
            # Warmup phase
            lr = self.base_lr * (global_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        return lr


class FocalLoss(nn.Cell):
    """
    Focal Loss for handling class imbalance
    Focuses training on hard examples
    """
    
    def __init__(self, alpha=None, gamma=2.0, num_classes=config.NUM_CLASSES):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        
        if alpha is None:
            self.alpha = Tensor(np.ones(num_classes), mindspore.float32)
        else:
            self.alpha = Tensor(alpha, mindspore.float32)
        
        self.softmax = nn.Softmax(axis=1)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
    
    def construct(self, logits, labels):
        """Calculate focal loss"""
        # Get probabilities
        probs = self.softmax(logits)
        
        # Get class probabilities
        class_mask = nn.OneHot(depth=self.num_classes)(labels)
        probs = (probs * class_mask).sum(axis=1)
        
        # Calculate focal term
        focal_weight = (1 - probs) ** self.gamma
        
        # Calculate cross entropy
        ce_loss = self.ce(logits, labels)
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


class MixedLoss(nn.Cell):
    """Combined loss: Focal Loss + Label Smoothing"""
    
    def __init__(self, class_weights=None, gamma=2.0, smoothing=0.1):
        super(MixedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=gamma)
        self.smoothing = smoothing
        self.num_classes = config.NUM_CLASSES
    
    def construct(self, logits, labels):
        """Calculate mixed loss"""
        # Focal loss
        focal = self.focal_loss(logits, labels)
        
        # Label smoothing
        confidence = 1.0 - self.smoothing
        smooth_labels = nn.OneHot(depth=self.num_classes)(labels)
        smooth_labels = smooth_labels * confidence + self.smoothing / self.num_classes
        
        log_probs = nn.LogSoftmax(axis=1)(logits)
        smooth_loss = -(smooth_labels * log_probs).sum(axis=1).mean()
        
        # Combine losses
        return 0.7 * focal + 0.3 * smooth_loss


def train_model(model_type='resnet', epochs=config.EPOCHS, batch_size=config.BATCH_SIZE):
    """
    Main training function
    
    Args:
        model_type: Type of model ('resnet' or 'efficientnet')
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    # Set MindSpore context
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=config.MINDSPORE_DEVICE_TARGET,
        device_id=config.MINDSPORE_DEVICE_ID
    )
    
    print("=" * 80)
    print(f"Training Skin Lesion Classifier with MindSpore")
    print(f"Model Type: {model_type}")
    print(f"Device: {config.MINDSPORE_DEVICE_TARGET}")
    print("=" * 80)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset, test_dataset, class_weights = create_simple_dataset(
        config.DATA_DIR,
        config.METADATA_FILE,
        is_training=True
    )
    
    train_steps = train_dataset.get_dataset_size()
    print(f"Training steps per epoch: {train_steps}")
    
    # Create model
    print(f"\nCreating {model_type} model...")
    network = create_model(model_type=model_type, num_classes=config.NUM_CLASSES)
    
    # Define loss function with class weights
    loss_fn = MixedLoss(class_weights=class_weights, gamma=2.0, smoothing=0.1)
    
    # Learning rate scheduler
    total_steps = train_steps * epochs
    lr_schedule = LearningRateScheduler(
        base_lr=config.LEARNING_RATE,
        total_steps=total_steps,
        warmup_steps=train_steps * 2,  # 2 epochs warmup
        min_lr=config.MIN_LR
    )
    
    # Optimizer with weight decay
    optimizer = nn.Adam(
        network.trainable_params(),
        learning_rate=lr_schedule,
        weight_decay=1e-4
    )
    
    # Define metrics
    metrics = {
        'Accuracy': Accuracy(),
    }
    
    # Create model
    model = Model(
        network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics
    )
    
    # Callbacks
    checkpoint_config = CheckpointConfig(
        save_checkpoint_steps=train_steps,
        keep_checkpoint_max=5
    )
    
    checkpoint_callback = ModelCheckpoint(
        prefix=f"skin_lesion_{model_type}",
        directory=config.CHECKPOINT_DIR,
        config=checkpoint_config
    )
    
    loss_monitor = LossMonitor(per_print_times=train_steps // 4)
    
    metrics_callback = MetricsCallback(
        eval_dataset=val_dataset,
        model=model,
        log_file=f'training_{model_type}.log'
    )
    
    early_stopping = EarlyStoppingCallback(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=0.001
    )
    
    callbacks = [
        checkpoint_callback,
        loss_monitor,
        metrics_callback,
        early_stopping
    ]
    
    # Train model
    print("\nStarting training...")
    print("=" * 80)
    
    start_time = time.time()
    
    model.train(
        epochs,
        train_dataset,
        callbacks=callbacks,
        dataset_sink_mode=False
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best validation accuracy: {metrics_callback.best_acc:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = model.eval(test_dataset)
    
    print("\nTest Results:")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Save final model
    final_model_path = os.path.join(config.CHECKPOINT_DIR, f'final_{model_type}.ckpt')
    save_checkpoint(network, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    return model, metrics_callback.history


def resume_training(checkpoint_path, model_type='resnet', epochs=10):
    """
    Resume training from a checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Type of model
        epochs: Additional epochs to train
    """
    # Set context
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=config.MINDSPORE_DEVICE_TARGET
    )
    
    # Create model
    network = create_model(model_type=model_type)
    
    # Load checkpoint
    param_dict = load_checkpoint(checkpoint_path)
    load_param_into_net(network, param_dict)
    print(f"Loaded checkpoint from: {checkpoint_path}")
    
    # Continue training
    return train_model(model_type=model_type, epochs=epochs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train skin lesion classifier')
    parser.add_argument('--model', type=str, default='resnet',
                       choices=['resnet', 'efficientnet'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    if args.resume:
        # Resume training
        model, history = resume_training(
            args.resume,
            model_type=args.model,
            epochs=args.epochs
        )
    else:
        # Train from scratch
        model, history = train_model(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    print("\nTraining complete!")
