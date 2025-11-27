import tensorflow as tf
import numpy as np
import math

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.d_model = d_model
        self.max_len = max_len

        # Create positional encoding
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        # Shape (1, max_len, d_model)
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, x, training=False):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        # Slice pe to the current sequence length
        seq_len = tf.shape(x)[1]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x, training=training)

class TransformerEncoderLayer(tf.keras.layers.Layer):
    """
    Replicates torch.nn.TransformerEncoderLayer with batch_first=True, norm_first=False
    """
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
        # Self Attention
        # Post-LN: x = norm(x + sublayer(x))
        src2 = self.self_attn(src, src, return_attention_scores=False)
        src = src + self.dropout1(src2, training=training)
        src = self.norm1(src)
        
        # Feed Forward
        src2 = self.linear2(self.dropout(self.linear1(src), training=training))
        src = src + self.dropout2(src2, training=training)
        src = self.norm2(src)
        return src

class TransformerClassifier(tf.keras.Model):
    def __init__(self, num_classes=2, d_model=28, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection: (B, 1, 64, T) -> (B, T, 64) -> (B, T, d_model)
        self.input_proj = tf.keras.layers.Dense(d_model, use_bias=False)
        self.input_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=512, dropout=dropout)
        
        # Transformer encoder layers
        self.enc_layers = [
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                activation='relu'
            ) for _ in range(num_layers)
        ]
        
        # Simple classifier head
        self.classifier_head = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_classes)
        ])
        
    def call(self, x, training=False):
        """
        Args:
            x: Input tensor of shape (B, 1, 64, T)
        Returns:
            Output tensor of shape (B, num_classes)
        """
        # Reshape: (B, 1, 64, T) -> (B, 64, T)
        x = tf.squeeze(x, axis=1)
        # Permute: (B, 64, T) -> (B, T, 64)
        x = tf.transpose(x, perm=[0, 2, 1])
        
        # Project to d_model: (B, T, 64) -> (B, T, d_model)
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # Add positional encoding
        x = self.pos_encoding(x, training=training)
        
        # Transformer encoding
        for layer in self.enc_layers:
            x = layer(x, training=training)
        
        # Global mean pooling over time: (B, T, d_model) -> (B, d_model)
        x = tf.reduce_mean(x, axis=1)
        
        # Classifier: (B, d_model) -> (B, num_classes)
        x = self.classifier_head(x, training=training)
        
        return x

def count_parameters(model):
    """Count trainable parameters"""
    # Note: Model must be built (called once) before counting
    return model.count_params()

def test_model_initialization():
    """Test that model produces different outputs for different inputs"""
    model = TransformerClassifier()
    
    # Create two different random inputs
    x1 = tf.random.normal((1, 1, 64, 128))
    x2 = tf.random.normal((1, 1, 64, 128))
    
    # Call once to build
    out1 = model(x1, training=False)
    out2 = model(x2, training=False)
    
    # Check if outputs are different
    diff = tf.reduce_max(tf.abs(out1 - out2))
    if diff < 1e-5:
        print("❌ ERROR: Model outputs are identical!")
        return False
    else:
        print("✓ Model produces different outputs for different inputs")
        print(f" Output 1: {out1.numpy()}")
        print(f" Output 2: {out2.numpy()}")
        print(f" Difference: {diff.numpy():.6f}")
        return True

if __name__ == '__main__':
    # Optimized configuration
    TEST_D_MODEL = 28
    TEST_NUM_HEADS = 4
    TEST_NUM_LAYERS = 2
    TEST_DROPOUT = 0.2
    TEST_NUM_CLASSES = 2
    TEST_MAX_PARAMS = 17000
    
    model = TransformerClassifier(
        num_classes=TEST_NUM_CLASSES,
        d_model=TEST_D_MODEL,
        num_heads=TEST_NUM_HEADS,
        num_layers=TEST_NUM_LAYERS,
        dropout=TEST_DROPOUT
    )
    
    # Build the model with a dummy input
    dummy = tf.random.normal((1, 1, 64, 128))
    _ = model(dummy)
    
    params = count_parameters(model)
    print(f'Total parameters: {params:,}')
    print(f'Parameter budget: {TEST_MAX_PARAMS:,}')
    
    if params <= TEST_MAX_PARAMS:
        print(f'✓ Within budget! ({TEST_MAX_PARAMS - params:,} parameters remaining)')
    else:
        print(f'❌ Exceeds budget by {params - TEST_MAX_PARAMS:,} parameters')
    
    # Test forward pass with fixed length
    print('\n--- Testing Fixed Length Input ---')
    x_fixed = tf.random.normal((2, 1, 64, 128))
    out = model(x_fixed)
    print(f'Input shape: {x_fixed.shape}')
    print(f'Output shape: {out.shape}')
    
    # Test forward pass with variable length
    print('\n--- Testing Variable Length Input ---')
    x_var = tf.random.normal((2, 1, 64, 200))
    out_var = model(x_var)
    print(f'Input shape: {x_var.shape}')
    print(f'Output shape: {out_var.shape}')
    
    # Test initialization
    print('\n--- Testing Model Initialization ---')
    test_model_initialization()
    
    # Check that different batches produce different outputs
    print("\n--- Testing Batch Outputs ---")
    out = model(x_fixed, training=False)
    print(f"Batch output 0: {out[0].numpy()}")
    print(f"Batch output 1: {out[1].numpy()}")
    
    is_diff = not np.allclose(out[0], out[1], atol=1e-5)
    print(f"Outputs are different: {is_diff}")