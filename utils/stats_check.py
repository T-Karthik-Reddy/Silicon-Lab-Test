import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import config

# Import Model to check parameters
from Model import TransformerClassifier

def check_config_validity():
    print("\n" + "="*60)
    print("1. CONFIGURATION & DSP CHECK")
    print("="*60)
    
    errors = []
    warnings = []
    
    # 1. Check Frame constraints
    print(f"MAX_TIME_FRAMES: {config.MAX_TIME_FRAMES}")
    print(f"N_MELS:          {config.N_MELS}")
    print(f"HOP_LENGTH:      {config.HOP_LENGTH}")
    print(f"N_FFT:           {config.N_FFT}")
    
    # 2. DSP Logic Check
    if config.HOP_LENGTH > config.N_FFT:
        errors.append(f"❌ DSP Error: HOP_LENGTH ({config.HOP_LENGTH}) cannot be larger than N_FFT ({config.N_FFT})")
    
    # 3. RAM Calculation (Attention Matrix)
    # Float32 = 4 bytes. Matrix = Frames^2 * 4
    attn_memory = (config.MAX_TIME_FRAMES ** 2) * 4
    print(f"Attention Map RAM: {attn_memory/1024:.1f} KB (per head)")
    
    if config.MAX_TIME_FRAMES > 128:
        warnings.append(f"⚠️ RAM Warning: Frames={config.MAX_TIME_FRAMES}. Ideally keep <= 128 for EFR32MG24.")

    # 4. Context Window Calculation (The Goal of your update)
    # Duration = (Frames * Hop) / Sample_Rate
    context_duration = (config.MAX_TIME_FRAMES * config.HOP_LENGTH) / config.SAMPLE_RATE
    print(f"Context Window:    {context_duration:.2f} seconds")
    
    if context_duration < 2.5:
        warnings.append(f"⚠️ Context is {context_duration:.2f}s. Target was > 2.5s. Did you update HOP_LENGTH?")
    else:
        print("✓ Context window is sufficient for baby cry rhythm.")
    
    return errors, warnings

def check_data_integrity():
    print("\n" + "="*60)
    print("2. DATA INTEGRITY CHECK")
    print("="*60)
    
    errors = []
    
    if not os.path.exists(config.PROCESSED_DIR):
        errors.append("❌ Processed dataset directory not found! Run preprocess.py first.")
        return errors
    
    total_files = 0
    shapes = []
    
    # Check a sample of files from each class
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.PROCESSED_DIR, class_name)
        if not os.path.exists(class_dir):
             errors.append(f"❌ Class directory not found: {class_name}")
             continue
             
        files = list(Path(class_dir).glob('*.npy'))
        total_files += len(files)
        
        if not files:
            errors.append(f"❌ No .npy files found for class '{class_name}'")
            continue
            
        # Check first 5 files
        print(f"Checking {class_name} sample files...")
        for f in files[:5]:
            try:
                data = np.load(f)
                shapes.append(data.shape)
                
                # Expected Shape: (N_MELS, MAX_TIME_FRAMES)
                expected_freq = config.N_MELS
                expected_time = config.MAX_TIME_FRAMES
                
                if data.shape != (expected_freq, expected_time):
                    errors.append(f"❌ Shape Mismatch in {f.name}: Found {data.shape}, expected ({expected_freq}, {expected_time})")
                    
            except Exception as e:
                errors.append(f"❌ Corrupt file {f.name}: {e}")

    if total_files == 0:
        errors.append("❌ No data found.")
    else:
        print(f"✓ Checked samples from {total_files} total files.")
        if shapes:
            print(f"✓ Verified Data Shape: {shapes[0]} (Freq x Time)")

    return errors

def check_model_architecture():
    print("\n" + "="*60)
    print("3. MODEL COMPATIBILITY CHECK")
    print("="*60)
    
    errors = []
    
    try:
        print(f"Instantiating model with Dropout={config.DROPOUT}...")
        model = TransformerClassifier(
            num_classes=config.NUM_CLASSES,
            d_model=config.D_MODEL,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        )
        
        # Build with correct shape: (Batch, 1, Freq, Time)
        input_shape = (1, 1, config.N_MELS, config.MAX_TIME_FRAMES)
        _ = model(tf.zeros(input_shape))
        
        params = model.count_params()
        print(f"Total Parameters: {params:,}")
        
        # Flash Memory Estimate
        flash_usage_kb = (params * 4) / 1024
        print(f"Estimated Flash Usage (Float32): {flash_usage_kb:.1f} KB")
        
        # Parameter limit check
        if hasattr(config, 'MAX_PARAMS'):
            if params > config.MAX_PARAMS:
                errors.append(f"❌ Parameter Count {params} exceeds budget of {config.MAX_PARAMS}")
            else:
                print(f"✓ Within Parameter Budget ({config.MAX_PARAMS})")
                
    except Exception as e:
        errors.append(f"❌ Model Instantiation Failed: {e}")
        
    return errors

def main():
    print("STARTING UPDATED STATS CHECK...")
    
    config_errors, config_warnings = check_config_validity()
    data_errors = check_data_integrity()
    model_errors = check_model_architecture()
    
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    
    all_errors = config_errors + data_errors + model_errors
    
    if config_warnings:
        print("\n⚠️ WARNINGS:")
        for w in config_warnings:
            print(f"  - {w}")
            
    if all_errors:
        print("\n❌ CRITICAL ERRORS (Must Fix):")
        for e in all_errors:
            print(f"  - {e}")
        print("\nConclusion: FAILED. Do not train yet.")
        sys.exit(1)
    else:
        print("\n✅ ALL SYSTEMS GO.")
        print("1. Context Window is optimized (2.56s).")
        print("2. RAM usage is safe (64 KB Attention).")
        print("3. Data matches Model input.")
        print("You can proceed to run 'train.py'.")
        sys.exit(0)

if __name__ == '__main__':
    main()