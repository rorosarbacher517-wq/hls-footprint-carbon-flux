# ChatGPT generated on Oct 9, 2025 

import tensorflow as tf
from tensorflow.keras import layers

class RadialConv2D(layers.Layer):
    """
    Radially symmetric 2D convolution: weight = w(r), same distance -> same weight.
    Supports both odd and even kernel sizes.
    """

    def __init__(self, filters, kernel_size, strides=1, padding="valid",
                 use_bias=True, **kwargs):
        super().__init__(**kwargs)

        # Handle kernel size
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        self.kh, self.kw = kh, kw
        self.filters = filters
        self.strides = (1, strides, strides, 1) if isinstance(strides, int) else (1, *strides, 1)
        self.padding = padding.upper()
        self.use_bias = use_bias

        # --- Compute "center" as kernel midpoint (supports even sizes) ---
        ci = (kh - 1) / 2.0
        cj = (kw - 1) / 2.0

        ii = tf.range(kh, dtype=tf.float32)[:, None]
        jj = tf.range(kw, dtype=tf.float32)[None, :]
        di = ii - ci
        dj = jj - cj
        r2 = tf.round(di**2 + dj**2)  # rounded squared distance
        r2_flat = tf.reshape(r2, [-1])
        uniq, idx = tf.unique(tf.cast(r2_flat, tf.int32))
        self.R = int(tf.shape(uniq)[0])
        self._radius_index = tf.reshape(idx, [kh, kw])

        self.radius_index = tf.Variable(
            self._radius_index, trainable=False, dtype=tf.int32
        )

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        self.in_ch = in_ch

        # Trainable weights per radius class per output channel
        self.w_radial = self.add_weight(
            name="w_radial",
            shape=(self.R, self.filters),
            initializer="he_normal",
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None

        super().build(input_shape)

    def call(self, x):
        # Build kernel from radial weights
        w_spatial_out = tf.gather(self.w_radial, self.radius_index)  # [kh,kw,filters]
        kernel = tf.repeat(w_spatial_out[:, :, None, :], repeats=self.in_ch, axis=2)
        y = tf.nn.conv2d(x, kernel, strides=self.strides, padding=self.padding)
        if self.use_bias:
            y = tf.nn.bias_add(y, self.bias)
        return y
