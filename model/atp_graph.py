
import tensorflow as tf
from model import losses


def build_graph():
    
    @tf.function(experimental_relax_shapes=True)
    def train_step(atp_model, optimizer, x, y, n_C, n_T, training=True, n_C_s = None, n_T_s = None, img_seg = False):
        with tf.GradientTape(persistent=True) as tape:

            μ, log_σ, y1 = atp_model([x, y, n_C, n_T, training, n_C_s, n_T_s])  
            if img_seg:
                likpp, mse = losses.categorical_ce(y1, μ)
            else:
                _, _, _, likpp, mse = losses.nll(y1, μ, log_σ)

        print('likpp', likpp.shape)
        gradients = tape.gradient(likpp, atp_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, atp_model.trainable_variables))
        return μ, log_σ, likpp, mse

    # tf.keras.backend.set_floatx('float32')
    return train_step
