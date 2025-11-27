import tensorflow as tf
import numpy as np
import math
import os
import config  # Imports your config.py to get correct N_MELS/FRAMES

# ==============================================================================
# 1. CONFIGURATION (Must match latest training)
# ==============================================================================
WEIGHTS_PATH = 'models/best_model.weights.h5' 
FLOAT_OUTPUT = 'audio_model_float32.tflite'
INT8_OUTPUT  = 'audio_model_int8.tflite'

# Model Hyperparameters (From your successful run)
H_NUM_CLASSES = 2
H_D_MODEL     = 28
H_NUM_HEADS   = 4
H_NUM_LAYERS  = 2
H_DROPOUT     = 0.1    # Updated to 0.1 based on your latest optimization
INPUT_SHAPE   = (1, 1, config.N_MELS, config.MAX_TIME_FRAMES) # (1, 1, 64, 128)

# ==============================================================================
# 2. MODEL DEFINITION (Exact Copy)
# ==============================================================================
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.d_model = d_model
        self.max_len = max_len
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x, training=training)

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation='relu'):
        super().__init__()
        self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model // nhead)
        self.linear1 = tf.keras.layers.Dense(dim_feedforward, activation=activation)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.linear2 = tf.keras.layers.Dense(d_model)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, src, training=False):
        src2 = self.self_attn(src, src, return_attention_scores=False)
        src = src + self.dropout1(src2, training=training)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.linear1(src), training=training))
        src = src + self.dropout2(src2, training=training)
        src = self.norm2(src)
        return src

class TransformerClassifier(tf.keras.Model):
    def __init__(self, num_classes=2, d_model=28, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = tf.keras.layers.Dense(d_model, use_bias=False)
        self.input_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.pos_encoding = PositionalEncoding(d_model, max_len=512, dropout=dropout)
        self.enc_layers = [
            TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 2,
                dropout=dropout, activation='relu'
            ) for _ in range(num_layers)
        ]
        self.classifier_head = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_classes)
        ])
        
    def call(self, x, training=False):
        x = tf.squeeze(x, axis=1)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.pos_encoding(x, training=training)
        for layer in self.enc_layers:
            x = layer(x, training=training)
        x = tf.reduce_mean(x, axis=1)
        x = self.classifier_head(x, training=training)
        return x

# ==============================================================================
# 3. CONCRETE FUNCTION (Required for Subclassed Models)
# ==============================================================================
def make_concrete_function(model, input_shape):
    @tf.function
    def serve_model(x):
        return model(x, training=False)
    input_spec = tf.TensorSpec(input_shape, tf.float32)
    return serve_model.get_concrete_function(input_spec)

# ==============================================================================
# 4. CONVERSION LOGIC
# ==============================================================================
def main():
    print("1. Rebuilding model architecture...")
    model = TransformerClassifier(
        num_classes=H_NUM_CLASSES, d_model=H_D_MODEL,
        num_heads=H_NUM_HEADS, num_layers=H_NUM_LAYERS, dropout=H_DROPOUT
    )
    
    # Initialize weights
    _ = model(tf.zeros(INPUT_SHAPE))

    print(f"2. Loading weights from {WEIGHTS_PATH}...")
    if not os.path.exists(WEIGHTS_PATH):
        print("❌ Error: Weights file not found!")
        return
    model.load_weights(WEIGHTS_PATH)
    print("   Weights loaded.")

    print("3. Tracing graph...")
    concrete_func = make_concrete_function(model, INPUT_SHAPE)

    # --- Convert to Float32 (Recommended for first run) ---
    print("\n--- Converting to Float32 ---")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    tflite_float = converter.convert()
    with open(FLOAT_OUTPUT, 'wb') as f:
        f.write(tflite_float)
    print(f"✅ Saved {FLOAT_OUTPUT}")

    # --- Convert to Int8 (For MVP Hardware Accelerator) ---
    print("\n--- Converting to Int8 ---")
    def representative_data_gen():
        for _ in range(100):
            data = np.random.normal(0, 1, INPUT_SHAPE).astype(np.float32)
            yield [data]

    converter_int8 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_data_gen
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.int8
    converter_int8.inference_output_type = tf.int8
    
    try:
        tflite_int8 = converter_int8.convert()
        with open(INT8_OUTPUT, 'wb') as f:
            f.write(tflite_int8)
        print(f"✅ Saved {INT8_OUTPUT}")
    except Exception as e:
        print(f"⚠️ Int8 Conversion Warning: {e}")

if __name__ == '__main__':
    main()