import tensorflow as tf
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import layers

# x = inputs[2]
# mask = ffp_mask = tf.cast(inputs[2]!=-9999.,tf.float32) 
# pool_size=2; strides=2; padding="VALID"; data_format="NCHW"
import tensorflow as tf

def masked_avg_pool2d2(x, mask, data_format="NHWC", pool_size=2, strides=2, padding="VALID"):
    """
    x, mask:
      - 4D: NHWC => (B, H, W, C)  or  NCHW => (B, C, H, W)
      - 5D: only NHWC-like supported: (B, T, H, W, C)
    
    data_format:
      - "NHWC" (default)  for 4D/5D
      - "NCHW"           for 4D only
    """
    rank = len(x.shape)
    if rank not in (4, 5):
        raise ValueError(f"Only 4D or 5D tensors supported, got rank={rank}")
    
    is_5d = (rank == 5)
    if is_5d and data_format != "NHWC":
        raise ValueError("5D support assumes channels-last: (B, T, H, W, C) with data_format='NHWC'.")
    
    # If 5D NHWC-like: (B, T, H, W, C) -> merge B and T
    if is_5d:
        B, T, H, W, C = x.shape
        x    = tf.reshape(x,   [B * T, H, W, C])
        mask = tf.reshape(mask,[B * T, H, W, C])
    
    mask = tf.cast(mask, x.dtype)

    # Parse kernel & stride
    if isinstance(pool_size, int):
        kh, kw = pool_size, pool_size
    else:
        kh, kw = pool_size
    
    if isinstance(strides, int):
        sh, sw = strides, strides
    else:
        sh, sw = strides
    
    # Build ksize/strides depending on data_format
    if data_format == "NHWC":
        # x: (N, H, W, C)  → pool over H,W only
        ksize    = [1, kh, kw, 1]
        strides_ = [1, sh, sw, 1]
    elif data_format == "NCHW":
        # x: (N, C, H, W)  → pool over H,W only
        ksize    = [1, 1, kh, kw]
        strides_ = [1, 1, sh, sw]
    else:
        raise ValueError(f"Unsupported data_format: {data_format}")
    
    area = tf.cast(kh * kw, x.dtype)
    
    # Masked pooling
    sum_x = tf.nn.avg_pool(
        x * mask, ksize=ksize, strides=strides_,
        padding=padding, data_format=data_format
    ) * area
    
    count = tf.nn.avg_pool(
        mask, ksize=ksize, strides=strides_,
        padding=padding, data_format=data_format
    ) * area
    
    y = tf.math.divide_no_nan(sum_x, count)
    mask_out = tf.cast(count > 0.0, x.dtype)
    
    # Reshape back to 5D if needed (NHWC-like only)
    if is_5d:
        # After pooling: (B*T, H2, W2, C)
        _, H2, W2, _ = y.shape
        y        = tf.reshape(y,        [B, T, H2, W2, C])
        mask_out = tf.reshape(mask_out, [B, T, H2, W2, C])
    
    return y, mask_out


def masked_avg_pool2d(x, mask, pool_size=2, strides=2, padding="VALID"):
    mask = tf.cast(mask, x.dtype)
    if isinstance(pool_size, int): 
        kh, kw = pool_size, pool_size
    else:                          
        kh, kw = pool_size
    
    if isinstance(strides, int):   
        sh, sw = strides, strides
    else:                          
        sh, sw = strides
    
    ksize    = [1, kh, kw, 1]
    strides_ = [1, sh, sw, 1]
    area = tf.cast(kh * kw, x.dtype)
    
    # Sum of valid values and count of valid pixels
    sum_x = tf.nn.avg_pool(x * mask, ksize=ksize, strides=strides_, padding=padding) * area
    count = tf.nn.avg_pool(mask,     ksize=ksize, strides=strides_, padding=padding) * area
    
    y = tf.math.divide_no_nan(sum_x, count)          # masked average
    mask_out = tf.cast(count > 0.0, x.dtype)         # any-valid in window
    return y, mask_out

class MaskedAveragePooling2D(layers.Layer):
    def __init__(self, pool_size=2, strides=2, padding="VALID", eps=1e-6, **kw):
        super().__init__(**kw)
        self.pool_size, self.strides, self.padding, self.eps = pool_size, strides, padding, eps
    def call(self, inputs):
        x, mask = inputs
        y, m = masked_avg_pool2d(x, mask, self.pool_size, self.strides, self.padding, self.eps)
        return [y, m]


