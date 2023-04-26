import tensorflow as tf
import math as m


def nll(real, pred, pred_log_sig, c, ϵ=0.001):
    pi = tf.constant(m.pi, dtype=tf.float32)
    real = tf.cast(real, tf.float32)
    μ, log_σ =  pred[:, c:], pred_log_sig[:, c:]
    mse_per_point = tf.math.square(tf.math.subtract(real, μ))
    lik_per_point = (1 / 2) * tf.math.divide(mse_per_point, tf.math.square(tf.math.exp(log_σ) + ϵ)) + tf.math.log(tf.math.exp(log_σ) + ϵ) + (1/2)*tf.math.log(2*pi)
    sum_lik = tf.math.reduce_sum(lik_per_point)
    
    return lik_per_point, sum_lik, tf.math.reduce_mean(lik_per_point), tf.math.reduce_mean(mse_per_point)
