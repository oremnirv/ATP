
import tensorflow as tf
from model import losses


def build_graph():
    
    @tf.function
    def train_step(atp_model, optimizer, x, y, n_C, training=True):

        with tf.GradientTape(persistent=True) as tape:

            pred = atp_model(x, training) 
            _, _, likpp, mse = losses.nll(y[:, n_C:], pred[:, :, 0], pred[:, :, 1], n_C)
        
        gradients = tape.gradient(likpp, atp_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, atp_model.trainable_variables))
        return pred, likpp, mse

    @tf.function
    def test_step(atp_model, x_te, y, n_C, training=False):

        pred_te = atp_model(x_te, training)
        _, _, likpp, mse  = losses.nll(y[:, n_C:], pred_te[:, :, 0], pred_te[:, :, 1],  n_C)
        return pred_te, likpp, mse

    tf.keras.backend.set_floatx('float32')
    return train_step, test_step
