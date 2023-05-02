import tensorflow as tf
from model import dot_prod

### Predicted value of Y = a constant and/or a weighted sum of one or more recent values of Y and/or a weighted sum of one or more recent values of the errors.

class ARIMA(tf.keras.Model):
    def __init__(self, p,d,q):
        super(ARIMA, self).__init__()
        self.p = p
        self.d = d
        self.q = q

        self.mu = tf.Variable(tf.random.normal([1], dtype=tf.float32))
        self.e = [tf.Variable(tf.random.normal([1], dtype=tf.float32))
                   for _ in range(q)]

        self.d0 = Dense(d)
        self.p0 = Dense(p)
        self.q0 = Dense(q)

    def call(self, input, training=True):
        
        y_hat  = self.mu + self.d0(input) - self.q0(self.e)
        

        σ = tf.exp(log_σ)
        if self.bound_std:

            σ = 0.01 + 0.99 * tf.math.softplus(log_σ)

        log_σ = tf.math.log(σ)
        return μ, log_σ
