
import tensorflow as tf
from model import losses


def build_graph():
    
    @tf.function(experimental_relax_shapes=True)
    def train_step(atp_model, optimizer, x, y, n_C, n_T, multiple = 1, training=True, y_re = None, n_C_s = None, n_T_s = None):

        with tf.GradientTape(persistent=True) as tape:

            μ, log_σ = atp_model([x, y, n_C, n_T, training, n_C_s, n_T_s]) 
            y1 = y[:, n_C_s * multiple:(n_T_s+n_C_s) * multiple]
            if y_re is not None:
                y1 = y_re
            _, _, _, likpp, mse = losses.nll(y1, μ, log_σ)
        
        gradients = tape.gradient(likpp, atp_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, atp_model.trainable_variables))
        return μ, log_σ, likpp, mse

    tf.keras.backend.set_floatx('float32')
    return train_step
