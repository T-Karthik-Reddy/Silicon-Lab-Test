import os
import sys
import numpy as np
import soundfile as sf
import scipy.signal
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf # Only used for TFLite Interpreter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config

# ==========================================
# HARDWARE-READY DSP FUNCTIONS (High Fidelity)
# ==========================================
def hz_to_mel(frequencies):
    return 2595.0 * np.log10(1.0 + frequencies / 700.0)

def mel_to_hz(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def create_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    fft_freqs = np.linspace(0, sr / 2, int(1 + n_fft // 2))
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    filters = np.zeros((n_mels, int(1 + n_fft // 2)))
    
    for i in range(n_mels):
        b_left = bin_points[i]
        b_center = bin_points[i + 1]
        b_right = bin_points[i + 2]
        
        for k in range(b_left, b_center):
            filters[i, k] = (k - b_left) / (b_center - b_left)
        for k in range(b_center, b_right):
            filters[i, k] = (b_right - k) / (b_right - b_center)
    return filters

def compute_stft_manual(y, n_fft, hop_length):
    pad_amount = n_fft // 2
    y_padded = np.pad(y, (pad_amount, pad_amount), mode='constant')
    n_frames = 1 + (len(y_padded) - n_fft) // hop_length
    window = np.hanning(n_fft)
    stft_matrix = []
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        frame = y_padded[start:end]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        windowed_frame = frame * window
        fft_frame = np.fft.rfft(windowed_frame, n=n_fft)
        stft_matrix.append(fft_frame)
    return np.array(stft_matrix).T

def power_to_db_manual(S, ref=1.0, amin=1e-10, top_db=80.0):
    S_log = 10.0 * np.log10(np.maximum(amin, S))
    S_log -= 10.0 * np.log10(np.maximum(amin, ref))
    if top_db is not None:
        S_log = np.maximum(S_log, S_log.max() - top_db)
    return S_log

def apply_preemphasis(y, coeff=0.97):
    return np.append(y[0], y[1:] - coeff * y[:-1])

# ==========================================
# INFERENCE CLASS
# ==========================================

class CryDetectorTFLite:
    def __init__(self, tflite_path, stats_path=None):
        """Initialize TFLite Cry Detector"""
        
        if not os.path.exists(tflite_path):
            raise FileNotFoundError(f"TFLite model not found: {tflite_path}")

        # 1. Load TFLite Model
        try:
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
            self.interpreter.allocate_tensors()
            
            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.input_index = self.input_details[0]['index']
            self.output_index = self.output_details[0]['index']
            self.input_shape = self.input_details[0]['shape']
            
            print(f"✓ TFLite Model loaded: {os.path.basename(tflite_path)}")
            print(f"  Input Shape Expected: {self.input_shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load TFLite interpreter: {e}")

        # 2. Load Normalization Stats
        # We need the stats used during training to normalize input
        stats = None
        if stats_path is None:
            # Look in the same directory as the model first
            stats_path = os.path.join(os.path.dirname(tflite_path), 'stats.npy')
            if not os.path.exists(stats_path):
                stats_path = os.path.join(config.MODEL_DIR, 'stats.npy')

        if os.path.exists(stats_path):
            try:
                stats = np.load(stats_path, allow_pickle=True).item()
                self.mean = stats['mean']
                self.std = stats['std']
                print(f"✓ Stats loaded (mean={self.mean:.2f}, std={self.std:.2f})")
            except:
                print("! Warning: Corrupt stats file. Using defaults.")
                self.mean, self.std = -50.0, 20.0
        else:
            print("! Warning: No stats file found. Using defaults.")
            self.mean, self.std = -50.0, 20.0

        # Pre-calculate Mel Basis (Optimization)
        self.mel_basis = create_mel_filterbank(
            config.SAMPLE_RATE, config.N_FFT, config.N_MELS, 50, 8000
        )

    def preprocess_audio(self, audio_path):
        """Hardware-compatible High Fidelity Preprocessing"""
        try:
            # 1. Load OGG/WAV (Polyphase Resampling)
            y, orig_sr = sf.read(audio_path, always_2d=False)
            if y.ndim > 1: y = np.mean(y, axis=1)
            
            if orig_sr != config.SAMPLE_RATE:
                gcd = np.gcd(orig_sr, config.SAMPLE_RATE)
                y = scipy.signal.resample_poly(y, config.SAMPLE_RATE//gcd, orig_sr//gcd)
            
            # 2. Pre-emphasis
            y = apply_preemphasis(y)
            
            # 3. Soft Silence Trim (-40dB)
            frame_len = 2048
            hop = 512
            y_sq = y**2
            energy = np.array([np.sum(y_sq[i:i+frame_len]) for i in range(0, len(y), hop)])
            if len(energy) > 0 and np.max(energy) > 0:
                thresh = np.max(energy) * (10**(-40/10))
                active = np.where(energy > thresh)[0]
                if len(active) > 0:
                    start = max(0, active[0]*hop - hop*2)
                    end = min(len(y), active[-1]*hop + frame_len + hop*2)
                    y = y[start:end]
            
            if len(y) == 0: return None, 0, y
            
            # 4. Manual STFT & Mel Spec
            stft_complex = compute_stft_manual(y, config.N_FFT, config.HOP_LENGTH)
            power_spec = np.abs(stft_complex)**2
            mel_spec = np.dot(self.mel_basis, power_spec)
            mel_spec_db = power_to_db_manual(mel_spec, ref=np.max(mel_spec))
            
            # 5. Pad/Truncate
            current_frames = mel_spec_db.shape[1]
            max_frames = config.MAX_TIME_FRAMES
            
            if current_frames < max_frames:
                pad_width = max_frames - current_frames
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='edge')
            else:
                start = (current_frames - max_frames) // 2
                mel_spec_db = mel_spec_db[:, start:start + max_frames]
                
            return mel_spec_db, len(y)/config.SAMPLE_RATE, y
            
        except Exception as e:
            print(f"Error preprocessing: {e}")
            return None, 0, None

    def predict(self, audio_path, visualize=False, save_viz=None):
        """Run TFLite Inference"""
        
        # 1. Preprocess
        mel_spec, duration, audio = self.preprocess_audio(audio_path)
        if mel_spec is None:
            raise ValueError("Audio preprocessing failed (empty or silent file)")

        # 2. Normalize
        mel_spec_norm = (mel_spec - self.mean) / (self.std + 1e-8)
        
        # 3. Reshape for TFLite
        # Input tensor is likely (1, 1, 64, 128) or (1, 64, 128)
        input_data = mel_spec_norm.astype(np.float32)
        
        # Add Batch Dimension
        input_data = input_data[np.newaxis, ...] 
        
        # Match dimensions dynamically
        # If model expects 4 dims (1, 1, 64, 128) and we have (1, 64, 128), add channel dim
        if len(self.input_shape) == 4 and len(input_data.shape) == 3:
            input_data = input_data[:, np.newaxis, ...] # Add channel dim at axis 1
        
        # Check if model expects (Batch, Time, Freq) vs (Batch, Freq, Time)
        # If model input matches input_data shape, great. If not, try transposing.
        if tuple(self.input_shape) != input_data.shape:
            # Try transposing last two dims
            input_data_T = np.transpose(input_data, (0, 1, 3, 2))
            if tuple(self.input_shape) == input_data_T.shape:
                input_data = input_data_T
            else:
                # Hard reshape if sizes match
                if np.prod(self.input_shape) == np.prod(input_data.shape):
                    input_data = input_data.reshape(self.input_shape)
                else:
                    raise ValueError(f"Shape mismatch. Model expects {self.input_shape}, got {input_data.shape}")

        # 4. Inference
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_index)
        
        # 5. Process Output
        # Output is logits or softmax depending on model export. 
        # We assume logits if values are not 0-1 sum.
        if np.any(output_data < 0) or np.sum(output_data) > 1.5:
            probs = tf.nn.softmax(output_data, axis=1).numpy()
        else:
            probs = output_data

        pred_class_idx = np.argmax(probs, axis=1)[0]
        confidence = probs[0][pred_class_idx]
        
        result = {
            'file': os.path.basename(audio_path),
            'duration': duration,
            'predicted_class': config.CLASS_NAMES[pred_class_idx],
            'confidence': confidence,
            'is_crying': config.CLASS_NAMES[pred_class_idx] == 'cry',
            'cry_probability': probs[0][0],
            'not_cry_probability': probs[0][1]
        }

        if visualize or save_viz:
            self._visualize(audio, mel_spec, result, save_viz)

        return result

    def _visualize(self, audio, mel_spec, result, save_path=None):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot Waveform
        times = np.arange(len(audio)) / config.SAMPLE_RATE
        axes[0].plot(times, audio, linewidth=0.5, color='#1f77b4')
        axes[0].set_title(f'Waveform - {result["file"]}')
        axes[0].grid(True, alpha=0.3)
        
        # Plot Mel Spec
        im = axes[1].imshow(mel_spec, aspect='auto', origin='lower', cmap='magma',
                           extent=[0, result['duration'], 0, config.N_MELS])
        axes[1].set_title('Mel Spectrogram (Hardware Input)')
        axes[1].set_ylabel('Mel Filter Bins')
        axes[1].set_xlabel('Time (s)')
        plt.colorbar(im, ax=axes[1], format='%+2.0f dB')
        
        # Text
        color = 'red' if result['is_crying'] else 'green'
        status = "CRY DETECTED" if result['is_crying'] else "No Cry Detected"
        info = f"{status}\nConf: {result['confidence']:.1%}\nCry Prob: {result['cry_probability']:.2f}"
        
        fig.text(0.5, 0.02, info, ha='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"✓ Viz saved: {save_path}")
        else:
            plt.show()
        plt.close()

def main():
    import argparse
    # Default TFLite path
    default_model = "audio_model_float32.tflite"
    
    parser = argparse.ArgumentParser(description='Baby Cry TFLite Inference')
    parser.add_argument('audio_files', nargs='+', help='Path to audio files')
    parser.add_argument('--model', default=default_model, help='Path to .tflite model')
    parser.add_argument('--visualize', '-v', action='store_true')
    parser.add_argument('--output', '-o', help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    try:
        detector = CryDetectorTFLite(args.model)
        
        if args.output: os.makedirs(args.output, exist_ok=True)
        
        print("\n" + "="*50)
        print("TFLite Inference Started")
        print("="*50)
        
        for f in args.audio_files:
            save_path = None
            if args.output:
                save_path = os.path.join(args.output, Path(f).stem + "_pred.png")
                
            res = detector.predict(f, visualize=args.visualize, save_viz=save_path)
            
            icon = "🔴" if res['is_crying'] else "🟢"
            print(f"{icon} {res['file']:<20} | {res['predicted_class'].upper()} ({res['confidence']:.1%})")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == '__main__':
    main()