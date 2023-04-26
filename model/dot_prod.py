###########################
# Author: Omer Nivron
###########################
import tensorflow as tf


def dot_product(q, k, v, mask):
    qk = tf.matmul(q, k, transpose_b=True)
    qk = tf.cast(qk, tf.float32)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    qk = tf.cast(qk / tf.math.sqrt(dk), tf.float32)
    if mask is not None:
        qk += ((tf.cast(mask[:, tf.newaxis, :, :], tf.float32)) * -1e9)

    w = tf.nn.softmax(qk, axis=-1, name='att_w')  # (batch_size X d_model X seq_len X seq_len)
    o = tf.matmul(w, tf.cast(v, tf.float32))  # (batch size, num heads, seq_len, 32)
    return o


class MultiHeadAttention2D(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention2D, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model, name='wq')
        self.wk = tf.keras.layers.Dense(d_model, name='wk')
        self.wv = tf.keras.layers.Dense(d_model, name='wv')
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_q, depth)

        o = dot_product(q, k, v, mask)
        o = tf.transpose(o, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        o = tf.reshape(o, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        o = self.dense(o)  # (batch_size, seq_len_q, d_model)
        return o



