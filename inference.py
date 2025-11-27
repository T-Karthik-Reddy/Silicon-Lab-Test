import os
import sys
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from Model import TransformerClassifier

class CryDetector:
    def __init__(self, model_path, stats_path=None):
        """Initialize the Transformer-based cry detector (TensorFlow)"""
        
        # Check availability of GPU
        gpus = tf.config.list_physical_devices('GPU')
        self.device = 'GPU' if gpus else 'CPU'
        print(f"Using device: {self.device}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Initialize Model Architecture
        self.model = TransformerClassifier(
            num_classes=config.NUM_CLASSES,
            d_model=config.D_MODEL,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        )
        
        # Build the model with a dummy input to create variables
        # Input shape matches dataset: (Batch, Channel, Freq, Time) -> (1, 1, 64, 128)
        dummy_input = tf.zeros((1, 1, 64, 128))
        _ = self.model(dummy_input, training=False)
        
        # Load weights
        try:
            self.model.load_weights(model_path)
            print(f"✓ Model weights loaded from {os.path.basename(model_path)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load weights: {e}")
        
        # Load normalization stats
        # In TF implementation, we saved meta_data.npy or stats.npy
        stats = None
        
        # 1. Try loading from metadata file (saved in Train.py)
        meta_path = os.path.join(os.path.dirname(model_path), 'meta_data.npy')
        if os.path.exists(meta_path):
            try:
                meta_data = np.load(meta_path, allow_pickle=True).item()
                stats = meta_data.get('stats')
                print(f"✓ Loaded metadata (Epoch {meta_data.get('epoch', 'unknown')})")
            except:
                pass
        
        # 2. Fallback to explicit stats.npy
        if stats is None:
            if stats_path is None:
                stats_path = os.path.join(config.MODEL_DIR, 'stats.npy')
            
            if os.path.exists(stats_path):
                stats = np.load(stats_path, allow_pickle=True).item()
            else:
                # Fallback for fresh inference without training artifacts
                print("! Warning: No stats file found. Using default normalization.")
                stats = {'mean': -50.0, 'std': 20.0}

        self.mean = stats['mean']
        self.std = stats['std']
        print(f"✓ Normalization stats loaded (mean={self.mean:.2f}, std={self.std:.2f})")
    
    def preprocess_audio(self, audio_path):
        """Load and preprocess audio file"""
        # Load audio with preprocessing
        y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
        
        # Get duration
        duration = len(y) / sr
        
        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Normalize audio
        if np.abs(y_trimmed).max() > 0:
            y_trimmed = y_trimmed / np.abs(y_trimmed).max()
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y_trimmed, 
            sr=config.SAMPLE_RATE, 
            n_mels=config.N_MELS, 
            n_fft=config.N_FFT, 
            hop_length=config.HOP_LENGTH,
            fmin=50,
            fmax=8000,
            power=2.0
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
        
        # Handle fixed length if specified in config (optional for Transformer, but good for consistency)
        if hasattr(config, 'MAX_TIME_FRAMES') and config.MAX_TIME_FRAMES is not None:
            current_frames = mel_spec_db.shape[1]
            if current_frames < config.MAX_TIME_FRAMES:
                pad_width = config.MAX_TIME_FRAMES - current_frames
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='edge')
            else:
                start = (current_frames - config.MAX_TIME_FRAMES) // 2
                mel_spec_db = mel_spec_db[:, start:start + config.MAX_TIME_FRAMES]
        
        return mel_spec_db, duration, y_trimmed
    
    def predict(self, audio_path, visualize=False, save_viz=None):
        """
        Predict if audio contains crying
        """
        # Check if file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"\nProcessing: {audio_path}")
        
        try:
            # Preprocess
            mel_spec, duration, audio = self.preprocess_audio(audio_path)
            print(f"✓ Audio loaded: {duration:.2f} seconds")
            
            # Normalize (Z-score)
            mel_spec_norm = (mel_spec - self.mean) / (self.std + 1e-8)
            
            # Prepare Tensor
            # 1. Add Channel dim: (F, T) -> (1, F, T)
            # 2. Add Batch dim: (1, F, T) -> (1, 1, F, T)
            mel_input = mel_spec_norm[np.newaxis, np.newaxis, ...]
            mel_tensor = tf.convert_to_tensor(mel_input, dtype=tf.float32)
            
            # Predict
            output = self.model(mel_tensor, training=False)
            probs = tf.nn.softmax(output, axis=1).numpy()
            pred_class = np.argmax(probs, axis=1)[0]
            confidence = probs[0][pred_class]
            
            # Get both class probabilities
            cry_prob = probs[0][0]
            not_cry_prob = probs[0][1]
            
            result = {
                'file': os.path.basename(audio_path),
                'duration': duration,
                'predicted_class': config.CLASS_NAMES[pred_class],
                'confidence': confidence,
                'is_crying': config.CLASS_NAMES[pred_class] == 'cry',
                'cry_probability': cry_prob,
                'not_cry_probability': not_cry_prob
            }
            
            # Visualize if requested
            if visualize or save_viz:
                self._visualize(audio, mel_spec, result, save_viz)
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            raise
    
    def _visualize(self, audio, mel_spec, result, save_path=None):
        """Visualize audio waveform and mel-spectrogram with prediction"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot waveform
        times = np.arange(len(audio)) / config.SAMPLE_RATE
        axes[0].plot(times, audio, linewidth=0.5)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'Audio Waveform - {result["file"]}')
        axes[0].grid(True, alpha=0.3)
        
        # Plot mel-spectrogram
        img = librosa.display.specshow(mel_spec, sr=config.SAMPLE_RATE, 
                                      hop_length=config.HOP_LENGTH,
                                      x_axis='time', y_axis='mel', ax=axes[1])
        axes[1].set_title('Mel-Spectrogram (Transformer Input)')
        plt.colorbar(img, ax=axes[1], format='%+2.0f dB')
        
        # Add prediction text
        prediction_text = f"Prediction: {result['predicted_class'].upper()}\n"
        prediction_text += f"Confidence: {result['confidence']:.1%}\n"
        prediction_text += f"Cry: {result['cry_probability']:.1%} | Not Cry: {result['not_cry_probability']:.1%}\n"
        prediction_text += f"Model: Transformer (TensorFlow)"
        
        # Color based on prediction
        color = 'red' if result['is_crying'] else 'green'
        fig.text(0.5, 0.02, prediction_text, ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def predict_batch(self, audio_files, output_dir=None):
        """Predict on multiple audio files"""
        results = []
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nProcessing {len(audio_files)} files...")
        print("="*60)
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] ", end="")
            
            try:
                save_viz = None
                if output_dir:
                    save_viz = os.path.join(output_dir, 
                                           f"{Path(audio_file).stem}_prediction.png")
                
                result = self.predict(audio_file, visualize=False, save_viz=save_viz)
                results.append(result)
                
                # Print result
                status = "🔴 CRYING" if result['is_crying'] else "🟢 NOT CRYING"
                print(f"✓ {status} ({result['confidence']:.1%} confidence)")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                results.append({
                    'file': os.path.basename(audio_file),
                    'error': str(e)
                })
        
        return results


def main():
    """Main function for command-line usage"""
    import argparse
    
    # Default weights path is different in TF (uses .weights.h5)
    default_model = os.path.join(config.MODEL_DIR, 'best_model.weights.h5')
    
    parser = argparse.ArgumentParser(description='Baby Cry Detection (TensorFlow Transformer)')
    parser.add_argument('audio_files', nargs='+', help='Audio file(s) to process')
    parser.add_argument('--model', default=default_model,
                       help='Path to model weights file')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Show visualization')
    parser.add_argument('--output-dir', '-o', help='Directory to save visualizations')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process multiple files')
    
    args = parser.parse_args()
    
    # Initialize detector
    try:
        detector = CryDetector(args.model)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Process files
    if args.batch or len(args.audio_files) > 1:
        # Batch processing
        results = detector.predict_batch(args.audio_files, args.output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        
        crying_count = sum(1 for r in results if r.get('is_crying', False))
        not_crying_count = sum(1 for r in results if not r.get('is_crying', True) and 'error' not in r)
        error_count = sum(1 for r in results if 'error' in r)
        
        print(f"Total files: {len(results)}")
        print(f"  🔴 Crying: {crying_count}")
        print(f"  🟢 Not Crying: {not_crying_count}")
        print(f"  ❌ Errors: {error_count}")
        
    else:
        # Single file processing
        audio_file = args.audio_files[0]
        save_viz = None
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            save_viz = os.path.join(args.output_dir, 
                                   f"{Path(audio_file).stem}_prediction.png")
        
        result = detector.predict(audio_file, visualize=args.visualize, save_viz=save_viz)
        
        # Print result
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"File: {result['file']}")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"\nPredicted Class: {result['predicted_class'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"\nClass Probabilities:")
        print(f"  Cry:     {result['cry_probability']:.1%}")
        print(f"  Not Cry: {result['not_cry_probability']:.1%}")
        print(f"\nIs Crying: {'YES 🔴' if result['is_crying'] else 'NO 🟢'}")
        print(f"\nModel: Transformer (TensorFlow)")
        print("="*60)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # Interactive mode
        print("="*60)
        print("Baby Cry Detector - Transformer Model (TensorFlow)")
        print("="*60)
        
        # Look for default weights
        model_path = os.path.join(config.MODEL_DIR, 'best_model.weights.h5')
        
        try:
            detector = CryDetector(model_path)
            
            while True:
                print("\nEnter audio file path (or 'quit' to exit):")
                audio_file = input("> ").strip()
                
                if audio_file.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not audio_file:
                    continue
                
                try:
                    result = detector.predict(audio_file, visualize=True)
                    
                    print("\n" + "-"*40)
                    print(f"Prediction: {result['predicted_class'].upper()}")
                    print(f"Confidence: {result['confidence']:.1%}")
                    print(f"Is Crying: {'YES 🔴' if result['is_crying'] else 'NO 🟢'}")
                    print("-"*40)
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
        except Exception as e:
            print(f"❌ Failed to initialize detector: {e}")
            print(f"Ensure 'best_model.weights.h5' exists in {config.MODEL_DIR}")
            sys.exit(1)
    else:
        # Command-line mode
        main()