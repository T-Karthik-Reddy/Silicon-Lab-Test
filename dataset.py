import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
import config

class CryDatasetGenerator:
    def __init__(self, file_paths, labels, stats, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.mean = stats['mean']
        self.std = stats['std']
        self.augment = augment
    
    def __len__(self):
        return len(self.file_paths)

    def time_mask(self, mel_spec, num_masks=2, mask_param=40):
        """Apply time masking (SpecAugment)"""
        mel_spec = mel_spec.copy()
        # mel_spec shape is (1, F, T)
        time_frames = mel_spec.shape[2]
        
        for _ in range(num_masks):
            t = np.random.randint(0, min(mask_param, time_frames))
            t0 = np.random.randint(0, time_frames - t)
            mel_spec[:, :, t0:t0+t] = 0
        return mel_spec
    
    def freq_mask(self, mel_spec, num_masks=2, mask_param=8):
        """Apply frequency masking (SpecAugment)"""
        mel_spec = mel_spec.copy()
        freq_bands = mel_spec.shape[1]
        
        for _ in range(num_masks):
            f = np.random.randint(0, min(mask_param, freq_bands))
            f0 = np.random.randint(0, freq_bands - f)
            mel_spec[:, f0:f0+f, :] = 0
        return mel_spec
    
    def add_noise(self, mel_spec, noise_factor=0.005):
        """Add random noise"""
        noise = np.random.randn(*mel_spec.shape) * noise_factor
        return mel_spec + noise
    
    def time_shift(self, mel_spec, shift_max=40):
        """Shift mel-spectrogram in time"""
        # mel_spec shape: (1, F, T)
        time_frames = mel_spec.shape[2]
        shift = np.random.randint(-shift_max, shift_max)
        
        if shift != 0:
            mel_spec = np.roll(mel_spec, shift, axis=2)
        return mel_spec
    
    def augment_data(self, mel_spec):
        """Apply random augmentations"""
        if np.random.random() < 0.5:
            mel_spec = self.time_mask(mel_spec, num_masks=1, mask_param=40)
        
        if np.random.random() < 0.5:
            mel_spec = self.freq_mask(mel_spec, num_masks=1, mask_param=8)
        
        if np.random.random() < 0.3:
            mel_spec = self.add_noise(mel_spec, noise_factor=0.003)
            
        if np.random.random() < 0.3:
            mel_spec = self.time_shift(mel_spec, shift_max=30)
            
        return mel_spec

    def __call__(self):
        """Generator function for tf.data.Dataset"""
        for idx in range(len(self.file_paths)):
            try:
                mel_spec = np.load(self.file_paths[idx])
                
                # Normalize
                mel_spec = (mel_spec - self.mean) / (self.std + 1e-8)
                
                # Add channel dimension: (F, T) -> (1, F, T)
                # Note: keeping NCHW format to match PyTorch logic exactly
                mel_spec = mel_spec[np.newaxis, ...]
                mel_spec = mel_spec.astype(np.float32)
                
                if self.augment:
                    mel_spec = self.augment_data(mel_spec)
                
                label = self.labels[idx]
                yield mel_spec, label
            except Exception as e:
                print(f"Error loading {self.file_paths[idx]}: {e}")
                continue

def load_data():
    """Load processed data and create train/val/test splits"""
    file_paths = []
    labels = []
    
    print("\nLoading dataset...")
    for idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(config.PROCESSED_DIR, class_name)
        
        if not os.path.exists(class_dir):
            print(f"ERROR: Directory not found: {class_dir}")
            raise FileNotFoundError(f"Directory not found: {class_dir}")
            
        files = list(Path(class_dir).glob('*.npy'))
        print(f" {class_name}: {len(files)} files")
        
        file_paths.extend([str(f) for f in files])
        labels.extend([idx] * len(files))
    
    if not file_paths:
        raise ValueError("No data files found!")
        
    # Split data
    try:
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            file_paths, labels, 
            test_size=(1-config.TRAIN_SPLIT), 
            random_state=42, 
            stratify=labels
        )
        
        val_size = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels, 
            test_size=(1-val_size), 
            random_state=42, 
            stratify=temp_labels
        )
    except ValueError:
        print("\nERROR during data splitting")
        raise
    
    # Calculate stats
    print("\nCalculating normalization statistics from training data...")
    train_values = []
    for f in train_files[:100]:
        train_values.append(np.load(f).flatten())
    train_values = np.concatenate(train_values)
    
    stats = {
        'mean': float(train_values.mean()), 
        'std': float(train_values.std())
    }
    
    print(f'Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}')
    print(f'Normalization stats - Mean: {stats["mean"]:.4f}, Std: {stats["std"]:.4f}')
    
    use_augment = hasattr(config, 'USE_AUGMENTATION') and config.USE_AUGMENTATION
    
    # Create generators
    train_gen = CryDatasetGenerator(train_files, train_labels, stats, augment=use_augment)
    val_gen = CryDatasetGenerator(val_files, val_labels, stats, augment=False)
    test_gen = CryDatasetGenerator(test_files, test_labels, stats, augment=False)
    
    return train_gen, val_gen, test_gen, stats

def get_dataloaders(batch_size=None):
    if batch_size is None:
        batch_size = config.BATCH_SIZE
        
    train_gen, val_gen, test_gen, stats = load_data()
    
    # Define output signature
    output_signature = (
        tf.TensorSpec(shape=(1, 64, None), dtype=tf.float32), # Variable time dimension
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )
    
    # Helper to create dataset
    def create_tf_dataset(generator, shuffle=False):
        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        
        # Padded batch allows variable length T in (1, 64, T)
        ds = ds.padded_batch(
            batch_size, 
            padded_shapes=((1, 64, None), ()),
            padding_values=(0.0, tf.constant(0, dtype=tf.int64))
        )
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    
    train_loader = create_tf_dataset(train_gen, shuffle=True)
    val_loader = create_tf_dataset(val_gen, shuffle=False)
    test_loader = create_tf_dataset(test_gen, shuffle=False)
    
    return train_loader, val_loader, test_loader, stats

if __name__ == '__main__':
    print("Testing dataset loading...")
    train_loader, val_loader, test_loader, stats = get_dataloaders()
    
    print("\nTesting data loading...")
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f" Data shape: {data.shape}")
        print(f" Labels shape: {labels.shape}")
        print(f" Labels: {labels[:5].numpy().tolist()}")
        print(f" Data range: [{np.min(data):.2f}, {np.max(data):.2f}]")
        
        if batch_idx == 2:
            break
            
    print("\nDataset loading test completed successfully!")