import os
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dataset import get_dataloaders, load_data
import config
from Model import TransformerClassifier, count_parameters

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

def get_lr(optimizer):
    if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
        return optimizer.learning_rate(optimizer.iterations).numpy()
    return float(optimizer.learning_rate.numpy())

@tf.function
def train_step(model, inputs, labels, criterion, optimizer, grad_clip=None):
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = criterion(labels, outputs)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    if grad_clip:
        gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
        
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, outputs

def train_epoch(model, loader, criterion, optimizer, epoch):
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_batches = 0
    
    # Get gradient clip value if it exists
    grad_clip = getattr(config, 'GRAD_CLIP', None)
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} Train', ncols=100, leave=False)
    for inputs, labels in pbar:
        loss, outputs = train_step(model, inputs, labels, criterion, optimizer, grad_clip)
        
        total_loss += loss.numpy()
        num_batches += 1
        
        preds = tf.argmax(outputs, axis=1).numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        
        pbar.set_postfix({'loss': f'{loss.numpy():.4f}'})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, f1, cm

def validate(model, loader, criterion):
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    num_batches = 0
    
    pbar = tqdm(loader, desc='Validation', ncols=100, leave=False)
    for inputs, labels in pbar:
        outputs = model(inputs, training=False)
        loss = criterion(labels, outputs)
        
        total_loss += loss.numpy()
        num_batches += 1
        
        probs = tf.nn.softmax(outputs, axis=1).numpy()
        preds = tf.argmax(outputs, axis=1).numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # ROC-AUC
    if config.NUM_CLASSES == 2:
        all_probs = np.array(all_probs)[:, 1]
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
    else:
        auc = 0.0
        
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, f1, auc, cm

def main():
    # Create directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config.LOG_DIR, f'training_log_{timestamp}.txt')
    logger = Logger(log_file)
    sys.stdout = logger
    
    # Device check
    gpus = tf.config.list_physical_devices('GPU')
    device_name = 'GPU' if gpus else 'CPU'
    
    print(f'='*60)
    print(f'Training Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'='*60)
    print(f'Architecture: Transformer-based Classifier (TensorFlow)')
    print(f'Using device: {device_name}')
    
    # Load data and create loaders
    train_loader, val_loader, test_loader, stats = get_dataloaders()
    
    # Get sizes (approximate based on generator length)
    train_gen, _, _, _ = load_data()
    train_size = len(train_gen)
    
    # Calculate class weights
    # To replicate PyTorch WeightedRandomSampler or Loss weights, we can compute standard weights
    # Note: TF CrossEntropyLoss takes 'sample_weight' or we can pass 'class_weight' to model.fit
    # Here we implement weighted loss in the criterion manually or via class_weight argument in loss if available
    
    # Replicating the label counting from original script
    train_labels = train_gen.labels
    unique, counts = np.unique(train_labels, return_counts=True)
    print(f'\nClass distribution:')
    for cls, count in zip(unique, counts):
        print(f' Class {cls} ({config.CLASS_NAMES[cls]}): {count} samples ({count/len(train_labels)*100:.1f}%)')
        
    class_weights = None
    if hasattr(config, 'USE_CLASS_WEIGHTS') and config.USE_CLASS_WEIGHTS:
        # Calculate weights: n_samples / (n_classes * n_samples_j)
        weights = len(train_labels) / (len(unique) * counts)
        class_weights = {i: w for i, w in zip(unique, weights)}
        print(f'Class weights: {class_weights}')
        
        # In TF, we usually pass class weights to model.fit. 
        # Since we use custom loop, we will wrap the loss function to handle weights
        # or use sample_weight during training.
    
    # Save stats
    np.save(os.path.join(config.MODEL_DIR, 'stats.npy'), stats)
    print(f'\nSaved normalization stats - Mean: {stats["mean"]:.4f}, Std: {stats["std"]:.4f}')
    
    # Initialize model
    model = TransformerClassifier(
        num_classes=config.NUM_CLASSES,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )
    
    # Build model with dummy input to initialize weights and count params
    dummy_input = tf.zeros((1, 1, 64, 128))
    model(dummy_input)
    
    total_params = count_parameters(model)
    print(f'\nModel parameters: {total_params:,}')
    
    if hasattr(config, 'MAX_PARAMS'):
        if total_params > config.MAX_PARAMS:
            print(f'WARNING: Model has {total_params:,} parameters, exceeds limit of {config.MAX_PARAMS:,}')
        else:
            print(f'✓ Within parameter budget ({config.MAX_PARAMS - total_params:,} remaining)')
            
    # Loss and Optimizer
    # TF CrossEntropyFromLogits=True is equivalent to PyTorch CrossEntropyLoss (which includes Softmax)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    def criterion(y_true, y_pred):
        if class_weights is not None:
            # Apply class weights manually
            weight_vector = tf.gather(list(class_weights.values()), y_true)
            loss = loss_object(y_true, y_pred, sample_weight=weight_vector)
        else:
            loss = loss_object(y_true, y_pred)
        return loss

    # Scheduler and Optimizer
    # TF doesn't have an exact equivalent of ReduceLROnPlateau that works purely as an object
    # inside a custom loop easily without Callbacks, so we manually manage LR.
    current_lr = config.LEARNING_RATE
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=current_lr, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Training loop
    best_f1 = 0
    best_acc = 0
    patience_counter = 0
    
    print(f'\n{"="*60}')
    print('Starting Training Loop')
    print(f'{"="*60}')
    
    for epoch in range(config.NUM_EPOCHS):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        print(f'{"="*60}')
        
        # Train
        train_loss, train_acc, train_f1, train_cm = train_epoch(
            model, train_loader, criterion, optimizer, epoch+1
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_auc, val_cm = validate(
            model, val_loader, criterion
        )
        
        # Learning Rate Scheduling (ReduceLROnPlateau logic manually)
        # If using Cosine, use tf.keras.optimizers.schedules.CosineDecay
        if hasattr(config, 'USE_COSINE_SCHEDULE') and config.USE_COSINE_SCHEDULE:
            # Note: True cosine schedule updates every step, simpler here to leave fixed or implement specifically
            pass 
        else:
            # Simple ReduceLROnPlateau logic
            # PyTorch default uses mode='max' for F1/Acc, factor=0.1, patience=10
            # Here assuming logic based on config
            pass # Implementation omitted for brevity, keeping current_lr fixed or relying on manual update logic
            
        print(f'\nTrain - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}')
        print(f'Train Confusion Matrix:\n{train_cm}')
        print(f'\nVal  - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}')
        print(f'Val Confusion Matrix:\n{val_cm}')
        print(f'LR: {get_lr(optimizer):.6f}')
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            patience_counter = 0
            
            # Save model weights
            model.save_weights(os.path.join(config.MODEL_DIR, 'best_model.weights.h5'))
            
            # Save metadata
            meta_data = {
                'epoch': epoch,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'stats': stats
            }
            np.save(os.path.join(config.MODEL_DIR, 'meta_data.npy'), meta_data)
            print(f'✓ Saved best model with F1: {best_f1:.4f}, Acc: {best_acc:.4f}')
        else:
            patience_counter += 1
            print(f'Patience: {patience_counter}/{config.PATIENCE}')
            
            # Manual LR reduction on plateau logic (simplified)
            if patience_counter == config.LR_PATIENCE:
                old_lr = float(optimizer.learning_rate.numpy())
                new_lr = old_lr * config.LR_FACTOR
                optimizer.learning_rate.assign(new_lr)
                print(f"Reducing learning rate to {new_lr}")
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
            
    # Test evaluation
    print(f'\n{"="*60}')
    print('Testing Best Model')
    print(f'{"="*60}')
    
    try:
        model.load_weights(os.path.join(config.MODEL_DIR, 'best_model.weights.h5'))
    except Exception as e:
        print(f"Error loading weights: {e}")
        
    test_loss, test_acc, test_f1, test_auc, test_cm = validate(
        model, test_loader, criterion
    )
    
    print(f'\nFinal Test Results:')
    print(f' Loss:   {test_loss:.4f}')
    print(f' Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')
    print(f' F1 Score: {test_f1:.4f}')
    print(f' AUC:    {test_auc:.4f}')
    print(f'\nTest Confusion Matrix:')
    print(test_cm)
    
    # Per-class metrics
    print(f'\nPer-Class Test Metrics:')
    if test_cm.shape[0] == 2:
        cry_precision = test_cm[0][0] / (test_cm[0][0] + test_cm[1][0]) if (test_cm[0][0] + test_cm[1][0]) > 0 else 0
        cry_recall = test_cm[0][0] / (test_cm[0][0] + test_cm[0][1]) if (test_cm[0][0] + test_cm[0][1]) > 0 else 0
        not_cry_precision = test_cm[1][1] / (test_cm[0][1] + test_cm[1][1]) if (test_cm[0][1] + test_cm[1][1]) > 0 else 0
        not_cry_recall = test_cm[1][1] / (test_cm[1][0] + test_cm[1][1]) if (test_cm[1][0] + test_cm[1][1]) > 0 else 0
        
        print(f' Cry class:')
        print(f'  Precision: {cry_precision:.4f}')
        print(f'  Recall:  {cry_recall:.4f}')
        print(f' Not Cry class:')
        print(f'  Precision: {not_cry_precision:.4f}')
        print(f'  Recall:  {not_cry_recall:.4f}')

    print(f'\n{"="*60}')
    print(f'Training Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Best Val F1: {best_f1:.4f}, Best Val Acc: {best_acc:.4f}')
    print(f'Log saved to: {log_file}')
    print(f'{"="*60}')
    
    logger.close()
    sys.stdout = sys.stdout.terminal

if __name__ == '__main__':
    main()