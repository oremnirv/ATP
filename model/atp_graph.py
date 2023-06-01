
import tensorflow as tf
from model import losses


def build_graph():
    
    @tf.function()
    def train_step(atp_model, optimizer, x, y, n_C, n_T, multiple = 1, training=True, n_C_s = None, n_T_s = None, n_C_tot = None, n_T_tot=None, subsample = False, bc = False, img_seg = False):
        with tf.GradientTape(persistent=True) as tape:

            μ, log_σ, y1 = atp_model([x, y, n_C, n_T, training, n_C_s, n_T_s, n_C_tot, n_T_tot])  
            if img_seg:
                likpp, mse = losses.categorical_ce(y1, μ)
            else:
                _, _, _, likpp, mse = losses.nll(y1, μ, log_σ)

        gradients = tape.gradient(likpp, atp_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, atp_model.trainable_variables))
        return μ, log_σ, likpp, mse

    # tf.keras.backend.set_floatx('float32')
    return train_step
