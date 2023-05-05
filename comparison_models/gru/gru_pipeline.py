import tensorflow as tf
from tensorflow import keras
from comparison_models.gru.gru import gru_model
import sys
sys.path.append("../../")
from data_wrangler.feature_extractor import feature_wrapper


class gru_pipeline(keras.models.Model):

    def __init__(self, rnn_units, permutation_repeats=1, 
                 num_layers=1):
        super().__init__()
        self._permutation_repeats = permutation_repeats
        self._feature_wrapper = feature_wrapper()
        self._gru = gru_model(rnn_units, num_layers)

    def call(self, inputs):

        x, y, n_C, n_T, training = inputs
        #x and y have shape batch size x length x dim

        x = x[:,:n_C+n_T,:]
        y = y[:,:n_C+n_T,:]
            
        x, y = self._feature_wrapper.permute([x, y, n_C, n_T, self._permutation_repeats]) ##### clean permute, and check permute target and/or context?

        gru_input = tf.concat([x,y],axis=2)
        μ, log_σ  = self._gru(gru_input, training)

        return μ[:, (-n_T-1):-1, :], log_σ[:, (-n_T-1):-1, :]
            

def instantiate_gru(dataset, training=True):
            
    if dataset == "weather":

        return gru_pipeline(rnn_units=[20, 20, 10], permutation_repeats=1, num_layers=3)
    

    # if dataset == "electricity":