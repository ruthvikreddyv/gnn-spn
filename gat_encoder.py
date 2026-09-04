"""
gat_encoder.py  -  Multi-Head GAT Encoder (NaN-safe, TF 2.6 compatible)
"""
import tensorflow as tf
import numpy as np


class MultiHeadGATLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim, n_heads=4, use_residual=True,
                 activation=tf.nn.relu, **kwargs):
        super().__init__(**kwargs)
        assert out_dim % n_heads == 0
        self.out_dim      = out_dim
        self.n_heads      = n_heads
        self.head_dim     = out_dim // n_heads
        self.activation   = activation
        self.use_residual = use_residual

        init = tf.keras.initializers.GlorotUniform()
        self.W = [
            tf.keras.layers.Dense(self.head_dim, use_bias=False,
                kernel_initializer=init, name=f"W_h{h}")
            for h in range(n_heads)
        ]
        self.a_vecs = [
            self.add_weight(shape=(2 * self.head_dim,),
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
                trainable=True, name=f"a_h{h}")
            for h in range(n_heads)
        ]
        if use_residual:
            self.res_proj = tf.keras.layers.Dense(
                out_dim, use_bias=False,
                kernel_initializer=init, name="res_proj")
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="ln")

    def call(self, inputs, training=False):
        H, A = inputs                           # (k, d_in), (k, k)
        k = tf.shape(H)[0]

        # Edge mask: include edges and self-loops
        edge_mask = tf.minimum(
            tf.cast(A > 0, tf.float32) + tf.eye(k), 1.0)  # (k,k)

        head_outs = []
        for h in range(self.n_heads):
            Z = self.W[h](H)                    # (k, head_dim)

            # Attention scores via matmul — avoids broadcast_to dynamic shape
            a_src = self.a_vecs[h][:self.head_dim]   # (head_dim,)
            a_dst = self.a_vecs[h][self.head_dim:]   # (head_dim,)

            # e_ij = LeakyReLU(a_src^T z_i + a_dst^T z_j)
            score_src = tf.matmul(Z, tf.expand_dims(a_src, 1))  # (k,1)
            score_dst = tf.matmul(Z, tf.expand_dims(a_dst, 1))  # (k,1)
            e = score_src + tf.transpose(score_dst)              # (k,k) broadcast
            e = tf.nn.leaky_relu(e, alpha=0.2)

            # Mask non-edges with -30 (safe: won't NaN in softmax)
            e = e * edge_mask + (1.0 - edge_mask) * (-30.0)
            e = tf.clip_by_value(e, -30.0, 30.0)

            alpha = tf.nn.softmax(e, axis=-1)   # (k,k)
            agg   = tf.matmul(alpha, Z)          # (k, head_dim)

            if self.activation is not None:
                agg = self.activation(agg)
            head_outs.append(agg)

        out = tf.concat(head_outs, axis=-1)     # (k, out_dim)
        if self.use_residual:
            out = out + self.res_proj(H)
        out = self.layer_norm(out)
        return out


class GATEncoder(tf.keras.Model):
    def __init__(self, n_layers=2, hidden_dim=64, out_dim=64,
                 n_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.gat_layers = []
        for l in range(n_layers):
            is_last = (l == n_layers - 1)
            self.gat_layers.append(
                MultiHeadGATLayer(
                    out_dim      = out_dim if is_last else hidden_dim,
                    n_heads      = n_heads,
                    use_residual = True,
                    activation   = None if is_last else tf.nn.relu,
                    name=f"gat_{l}")
            )

    def call(self, inputs, training=False):
        H, A = inputs
        for layer in self.gat_layers:
            H = layer((H, A), training=training)
        return H