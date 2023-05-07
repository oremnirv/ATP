###########################
# Author: Omer Nivron
###########################
import tensorflow as tf
from model import dot_prod


class MultiHeadAttention_new_block(tf.keras.layers.Layer):
    def __init__(self, num_heads, output_shape, projection_shape):
        super().__init__()
        self.attention = dot_prod.DotProductAttention()  # Scaled dot product attention
        self.heads = num_heads  # Number of attention heads to use
        self.projection_shape = projection_shape  # Dimensionality of the linearly projected queries, keys and values in normal MHA
        assert projection_shape % 2 == 0 

        self.W_q = tf.keras.layers.Dense(projection_shape//2)  # Learned projection matrix for the queries
        self.W_k = tf.keras.layers.Dense(projection_shape//2)  # Learned projection matrix for the keys
        self.W_v = tf.keras.layers.Dense(projection_shape//2)  # Learned projection matrix for the values
        self.W_qx = tf.keras.layers.Dense(projection_shape//2)  # Learned projection matrix for the x queries
        self.W_kx = tf.keras.layers.Dense(projection_shape//2)  # Learned projection matrix for the x keys
        self.W_vx = tf.keras.layers.Dense(projection_shape//2)  # Learned projection matrix for the x values
        self.W_o = tf.keras.layers.Dense(output_shape)  # Learned projection matrix for the multi-head output
        assert projection_shape % self.heads == 0
        assert num_heads % 2 == 0

        #heads must be a factor of projection_shape

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads,seq_length,-1)
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], heads, -1))
            x = tf.transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, projection_shape)
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], self.projection_shape))
        return x

    def call(self, queries, keys, values, query_x, keys_x, mask=None):


        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped_xy = self.reshape_tensor(self.W_q(queries), self.heads//2, True)
        q_reshaped_x = self.reshape_tensor(self.W_qx(query_x), self.heads//2, True)

        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped_xy = self.reshape_tensor(self.W_k(keys), self.heads//2, True)
        k_reshaped_x = self.reshape_tensor(self.W_kx(keys_x), self.heads//2, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped_xy = self.reshape_tensor(self.W_v(values), self.heads//2, True)
        v_reshaped_x = self.reshape_tensor(self.W_vx(values), self.heads//2, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped_xy = self.attention(q_reshaped_xy, k_reshaped_xy, v_reshaped_xy, self.projection_shape//2, mask)
        o_reshaped_x = self.attention(q_reshaped_x, k_reshaped_x, v_reshaped_x, self.projection_shape//2, mask)
        o_reshaped = tf.concat([o_reshaped_xy, o_reshaped_x], axis=1)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)

        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.W_o(output)


