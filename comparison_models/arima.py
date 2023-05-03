import tensorflow as tf
from model import dot_prod



class AR(tf.keras.Model):
    def __init__(self, num_ar_terms):
        super().__init__()
        # in literature num_ar_terms is called p
        self.num_ar_terms = num_ar_terms
        self.ar_terms = tf.keras.layers.Dense(num_ar_terms)

    def call(self, input, training=True):
        μ  = self.ar_terms(input) 
        return μ
    
class ARMA(tf.keras.Model):
    def __init__(self, num_ar_terms, num_err_terms):
        super().__init__()
         # in literature num_ar_terms is called p
         # in literature num_err_terms is called q
        self.num_ar_terms = num_ar_terms
        self.ar_terms = tf.keras.layers.Dense(num_ar_terms)

    def call(self, input, training=True):
        μ  = self.ar_terms(input) 
        return μ




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
        
        μ  = self.mu + self.d0(input) 
        σ = self.q0(self.e)
        log_σ = tf.math.log(σ)

        return μ, log_σ
