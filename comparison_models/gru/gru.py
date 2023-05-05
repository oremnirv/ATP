import tensorflow as tf

class gru_model(tf.keras.Model):
    def __init__(self, rnn_units, num_layers):
        super(gru_model, self).__init__()
        self.num_layers = num_layers
        self.grus = [tf.keras.layers.GRU(rnn_units[i],
                                   return_sequences=True, 
                                   return_state=True) for i in range(num_layers)]
        
        self.w1 = tf.keras.layers.Dense(1)
        self.w2 = tf.keras.layers.Dense(1)


    def call(self, inputs, training=False):
        #input shape: (batch_size, seq_length, features)
        x = inputs
        if len(x.shape) < 2:
            whole_seq = x[:, :, tf.newaxis] 
        elif len(x.shape) > 3:
            whole_seq = tf.squeeze(x, axis=-1)
        else:
            whole_seq = x

        
        for i in range(self.num_layers):
            whole_seq, _ = self.grus[i](whole_seq, training=training)
            print(whole_seq.shape)
        
        log_σ = self.w1(tf.nn.gelu(whole_seq))
        μ = self.w2(tf.nn.gelu(whole_seq))
        
        return μ, log_σ