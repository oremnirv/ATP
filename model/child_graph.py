
import tensorflow as tf
from model import losses


def build_graph():
    @tf.function
    def train_step(model, optimizer, x, y, training=True):
        with tf.GradientTape(persistent=True) as tape:

            # μ, log_σ = model(x, training)  
            μ = model(x, training=training)
            # log_σ = tf.cast(-11, tf.float32) * tf.ones_like(μ)
            # _, _, _, likpp, mse = losses.nll(y, μ)
            mse = losses.nll(y, μ)

        # gradients = tape.gradient(likpp, model.trainable_variables)
        gradients = tape.gradient(mse, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # return μ, log_σ, likpp, mse
        return μ, mse

    return train_step
