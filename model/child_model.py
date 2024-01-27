import tensorflow as tf


class CHILD(tf.keras.Model):
    def __init__(self, 
                  input_shape = 2048,
                  dropout_rate=0.
                  ):
        super().__init__()

        self.embedding = tf.keras.layers.Embedding(283, 10)
        self.w11 = tf.keras.layers.Dense(input_shape, activation='relu')
        self.w12 = tf.keras.layers.Dense(input_shape / 2 , activation='relu')
        self.w13 = tf.keras.layers.Dense(input_shape / 4, activation='relu')
        self.w14 = tf.keras.layers.Dense(input_shape / 8, activation='relu') 
        self.w15 = tf.keras.layers.Dense(input_shape / 8, activation='relu')
        self.w16 = tf.keras.layers.Dense(input_shape / 16, activation='relu')
        self.w18 = tf.keras.layers.Dense(32, activation='relu')
        self.w19 = tf.keras.layers.Dense(2)

        self.w21 = tf.keras.layers.Dense(input_shape, activation='relu')
        self.w22 = tf.keras.layers.Dense(input_shape / 2 , activation='relu')
        self.w23 = tf.keras.layers.Dense(input_shape / 4, activation='relu')
        self.w24 = tf.keras.layers.Dense(input_shape / 8, activation='relu') 
        self.w25 = tf.keras.layers.Dense(input_shape / 8, activation='relu')
        self.w26 = tf.keras.layers.Dense(input_shape / 16, activation='relu')
        self.w28 = tf.keras.layers.Dense(32, activation='relu')
        self.w29 = tf.keras.layers.Dense(2)
        self.e1 = tf.keras.layers.Dense(32)
        self.d = tf.keras.layers.Dropout(dropout_rate)
        self.b = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)

    def call(self, x, training=True):
        ## network with embedding
        batch_size = tf.shape(x)[0]
        x1 = self.b(x[:, :, 1:], training=training)
        X = tf.reshape(x1, (batch_size, -1))
        e = self.embedding(x[:, 0, 0])
        e1 = self.e1(e)
        l1 = self.w11(X)
        l1 =  self.d(l1, training=training)
        l2 = self.w12(l1)
        l2 =  self.d(l2, training=training)
        l3 = self.w13(l2)
        l3 =  self.d(l3, training=training)    
        l4 = self.w14(l3)
        l4 =  self.d(l4, training=training)
        l5 = self.w15(l4) 
        l5 =  self.d(l5, training=training)
        l6 = self.w16(l5) 
        l6 =  self.d(l6, training=training) 
        l8 = self.w18(l6)  + e1
        l8 =  self.d(l8, training=training)
        l9 = self.w19(l8)

        ## network without embedding
        l21 = self.w21(X)
        l21 =  self.d(l21, training=training)
        l22 = self.w22(l21)
        l22 =  self.d(l22, training=training)
        l23 = self.w23(l22)
        l23 =  self.d(l23, training=training)
        l24 = self.w24(l23)
        l24 =  self.d(l24, training=training)
        l25 = self.w25(l24)
        l25 =  self.d(l25, training=training)
        l26 = self.w26(l25)
        l26 =  self.d(l26, training=training)
        l28 = self.w28(l26)
        l28 =  self.d(l28, training=training)
        l29 = self.w29(l28)
        ###
        o = l29 + l9
        μ = o[:, 0:1]
        #log_σ = o[:, 1:2]

        # σ = tf.exp(log_σ)
        # log_σ = tf.math.log(σ)
    
        return μ #, log_σ
