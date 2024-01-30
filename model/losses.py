import tensorflow as tf
import math as m


def nll(y, μ, log_σ = 0, ϵ=0.001):
    pi = tf.constant(m.pi, dtype=tf.float32)
    y = tf.cast(y, tf.float32)
    mse_per_point = tf.math.square(tf.math.subtract(y, μ))
    lik_per_point = (1 / 2) * tf.math.divide(mse_per_point, tf.math.square(tf.math.exp(log_σ) + ϵ)) + tf.math.log(tf.math.exp(log_σ) + ϵ) + (1/2)*tf.math.log(2*pi)
    sum_lik = tf.math.reduce_sum(lik_per_point)
    sum_mse = tf.math.reduce_sum(mse_per_point)
    
    return lik_per_point, sum_mse, sum_lik, tf.math.reduce_mean(lik_per_point), tf.math.reduce_mean(mse_per_point)
    # return tf.math.reduce_mean(mse_per_point)

def sparse_cat(y, μ):
    y = tf.cast(y, tf.float32)
    sparse_cat = tf.keras.losses.SparseCategoricalCrossentropy()
    likpp = sparse_cat(y[:, :, :1], μ)
    mse = tf.zeros(1)
    return tf.math.reduce_mean(likpp), mse


def categorical_ce(y, logit, reduce_mean=True):
    y = y[:, :, :1]
    cce = \
        -tf.reduce_sum(
            tf.math.log(
                tf.nn.softmax(logit, axis=-1)
            ) *
            tf.squeeze(tf.one_hot(
                tf.cast(y, tf.int32), 
                logit.shape[2]
            )),
            axis=-1
        )
    if reduce_mean:
        cce = tf.reduce_mean(cce)
    mse = tf.zeros(1)
    return cce, mse