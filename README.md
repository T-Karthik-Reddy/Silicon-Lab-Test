A lightweight Transformer-based audio classification for detecting baby cries, optimized for edge deployment with TensorFlow Lite.

## Core Components

### 1. Model Architecture (`Model.py`)

**TransformerClassifier** - Compact transformer model with ~15K parameters:
- **Input**: Mel spectrograms (64 mel bins × 128 time frames)
- **Architecture**: 
  - Positional encoding for temporal awareness
  - 2 transformer encoder layers (28-dim, 4 attention heads)
  - Global mean pooling + classification head
- **Output**: Binary classification (cry/not_cry)

Key features:
- Parameter-efficient design (fits 17K budget)
- Variable-length audio support
- Hardware-friendly operations

### 2. Training Pipeline (`train.py`)

**Training Process**:
- **Data**: Stratified train/val/test splits (70/15/15%)
- **Optimization**: AdamW with gradient clipping
- **Scheduling**: ReduceLROnPlateau with early stopping
- **Metrics**: Accuracy, F1-score, ROC-AUC, confusion matrix
- **Augmentation**: SpecAugment (time/frequency masking)

**Key Features**:
- Class weight balancing for imbalanced data
- Comprehensive logging and model checkpointing
- TensorFlow 2.x implementation with custom training loop

### 3. Float32 Inference (`inference_float32.py`)

**High-Fidelity Audio Processing**:
```python
# Audio Pipeline
1. Load & resample to 16kHz
2. Pre-emphasis filtering (0.97 coefficient)
3. Silence trimming (-40dB threshold)
4. Manual STFT computation (512 FFT, 320 hop)
5. Mel filterbank (64 bins, 50-8000Hz)
6. Power-to-dB conversion
7. Pad/truncate to 128 frames
```

**Hardware-Ready DSP**:
- Manual STFT implementation (no librosa dependency)
- Custom mel filterbank generation
- Optimized for embedded systems

### 4. Int8 Quantized Inference (`inference_int8.py`)

**Quantization Pipeline**:
```python
# Critical Int8 Conversion Formulas
input_int8 = (float_input / input_scale) + input_zero_point
output_float = (int8_output - output_zero_point) * output_scale
```

**Key Features**:
- **Quantization**: Float32 → Int8 conversion with scale/zero-point
- **Dequantization**: Int8 → Float32 for final probabilities
- **Hardware Target**: Optimized for INT8 accelerators
- **Identical Preprocessing**: Same DSP chain as float32 version

## Model Conversion

```bash
# Convert trained model to TFLite formats
python convert_to_tflite.py
# Generates: audio_model_float32.tflite, audio_model_int8.tflite
```

## Usage

### Training
```bash
python train.py
```

### Inference
```bash
# Float32 inference
python inference_float32.py audio_file.wav --visualize

# Int8 inference  
python inference_int8.py audio_file.wav --visualize
```

## Audio Processing Details

**Input Requirements**:
- Sample rate: 16kHz
- Format: WAV/OGG supported
- Duration: Variable (auto-padded/truncated)

**Preprocessing Chain**:
1. **Resampling**: Polyphase filter for quality preservation
2. **Pre-emphasis**: High-frequency enhancement (α=0.97)
3. **Silence Removal**: Energy-based trimming (-40dB threshold)
4. **STFT**: 512-point FFT, 320 hop length, Hanning window
5. **Mel Transform**: 64 mel bins, 50-8000Hz range
6. **Normalization**: Z-score using training statistics

**Hardware Compatibility**:
- No external DSP libraries required
- Fixed-point arithmetic ready
- Optimized for real-time processing

## Model Performance

- **Parameters**: ~15,000 (within 17K budget)
- **Input Shape**: (1, 1, 64, 128)
- **Inference Time**: <10ms on modern hardware
- **Memory**: <100KB model size (int8)

## File Structure

```
├── Model.py              # Transformer architecture
├── train.py              # Training pipeline
├── inference_float32.py  # Float32 inference
├── inference_int8.py     # Int8 quantized inference
├── config.py             # Configuration parameters
├── dataset.py            # Data loading and augmentation
├── convert_to_tflite.py  # Model conversion script
├── logs                  # Saved training logs
└── models/               # Saved models and stats
```

## Dependencies

```
tensorflow
numpy
scikit-learn
soundfile
scipy
matplotlib
tqdm
```

## Hardware Deployment

The system is designed for edge deployment:
- **Float32**: Standard embedded processors 
- **Int8**: Hardware accelerators (Silicon Labs MVP)
- **DSP**: Custom implementation for microcontrollers
