
import tensorflow as tf
from model import losses


def build_graph():
    
    @tf.function
    def train_step(atp_model, optimizer, x, y, n_C, n_T, training=True):

        with tf.GradientTape(persistent=True) as tape:

            mu,log_sigma = atp_model([x, y, n_C, n_T, training]) 

            _, _, likpp, mse = losses.nll(y[:, n_C:n_T+n_C], mu, log_sigma)
        
        gradients = tape.gradient(likpp, atp_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, atp_model.trainable_variables))
        return mu, log_sigma,likpp, mse

    @tf.function
    def test_step(atp_model, x, y, n_C, n_T, training=False):

        mu,log_sigma = atp_model([x, y, n_C, n_T, training])        
        _, _, likpp, mse = losses.nll(y[:, n_C:n_T+n_C], mu, log_sigma)
        return  mu, log_sigma,likpp, mse

    tf.keras.backend.set_floatx('float32')
    return train_step, test_step
