import tensorflow as tf
from tensorflow import keras
import numpy as np
from data_wrangler.feature_extractor import  DE, feature_wrapper
from model.atp import ATP
from model.atp_no_leakage import ATP as ATP_no_leakage
from model.atp_no_leakage_new_block import ATP as ATP_new_block
from model.atp_no_leakage_xxx import ATP as ATP_no_leakage_xxx



class atp_pipeline(keras.models.Model):
    
    def __init__(self, num_heads=4, projection_shape_for_head=4, output_shape=64, rate=0.1, permutation_repeats=1,
                 bound_std=False, num_layers=3, enc_dim=32, xmin=0.1, xmax=2, MHAX_leakage=True,**kwargs):
        super().__init__(**kwargs)
        # for testing set permutation_repeats=0
   
        self._permutation_repeats = permutation_repeats
        self.enc_dim = enc_dim
        self.xmin = xmin
        self.xmax = xmax
        self._feature_wrapper = feature_wrapper()
        if MHAX_leakage == True:
            self._atp = ATP(num_heads=num_heads,dropout_rate=rate,num_layers=num_layers,output_shape=output_shape,
                        projection_shape=projection_shape_for_head*num_heads,bound_std=bound_std)
        elif MHAX_leakage == False:
            self._atp = ATP_no_leakage(num_heads=num_heads,dropout_rate=rate,num_layers=num_layers,output_shape=output_shape,
                        projection_shape=projection_shape_for_head*num_heads,bound_std=bound_std)
        elif MHAX_leakage == "new_block":
            self._atp = ATP_new_block(num_heads=num_heads,dropout_rate=rate,num_layers=num_layers,output_shape=output_shape,
                        projection_shape=projection_shape_for_head*num_heads,bound_std=bound_std)
        elif MHAX_leakage == "xxx":
            self._atp = ATP_no_leakage_xxx(num_heads=num_heads,dropout_rate=rate,num_layers=num_layers,output_shape=output_shape,
                        projection_shape=projection_shape_for_head*num_heads,bound_std=bound_std)
        self._DE = DE()


    def call(self,inputs):

        x, y, n_C, n_T, training = inputs
        #x and y have shape batch size x length x dim

        x = x[:,:n_C+n_T,:]
        y = y[:,:n_C+n_T,:]

        if training == True:    
            x,y = self._feature_wrapper.permute([x, y, n_C, n_T, self._permutation_repeats]) ##### clean permute, and check permute target and/or context?
        
        x_emb = [self._feature_wrapper.PE([x[:, :, i][:, :, tf.newaxis], self.enc_dim, self.xmin, self.xmax]) for i in range(x.shape[-1])] 
        x_emb = tf.concat(x_emb, axis=-1)

        ######## make mask #######
        
        context_part = tf.concat([tf.ones((n_C,n_C),tf.bool),tf.zeros((n_C,n_T),tf.bool)],axis=-1)
        diagonal_mask = tf.linalg.band_part(tf.ones((n_C+n_T,n_C+n_T),tf.bool),-1,0)
        lower_diagonal_mask = tf.linalg.set_diag(diagonal_mask,tf.zeros(diagonal_mask.shape[0:-1],tf.bool)) ### condense into one line?                                                                               
        mask = tf.concat([context_part,lower_diagonal_mask[n_C:n_C+n_T,:n_C+n_T]],axis=0) # check no conflicts with init and check mask is correct shape
        
        ######## create derivative ########


        y_diff, x_diff, d, x_n, y_n = self._DE([y, x, n_C, n_T, training])

        inputs_for_processing = [x_emb, y, y_diff, x_diff, d, x_n, y_n, n_C, n_T]

        query_x, key_x, value_x, query_xy, key_xy, value_xy = self._feature_wrapper(inputs_for_processing)
        
        y_n_closest = y_n[:, :, :y.shape[-1]] #### need to update this based on how we pick closest point

        μ, log_σ = self._atp([query_x, key_x, value_x, query_xy, key_xy, value_xy, mask, y_n_closest],training=training)
        # print(μ[:, n_C])
        # print(μ[:, n_C].shape)

        return μ[:, n_C:], log_σ[:, n_C:]
      

def instantiate_atp(dataset,training=True):
    if dataset == "ETT":
        # return atp_pipeline(num_heads=4, projection_shape_for_head=4, output_shape=64, rate=0.1, permutation_repeats=1,
        #             bound_std=False, num_layers=2, enc_dim=32, xmin=0.1, xmax=2)

        return atp_pipeline(num_heads=6, projection_shape_for_head=11, output_shape=32, rate=0.05, permutation_repeats=0,
                 bound_std=False, num_layers=4, enc_dim=32, xmin=0.1, xmax=1,MHAX_leakage="xxx")      

    # elif dataset == "traffic":
    #     return atp_pipeline(num_heads=4, projection_shape_for_head=4, output_shape=64, rate=0.1, permutation_repeats=1,
    #                 bound_std=False, num_layers=2, enc_dim=32, xmin=0.1, xmax=2)
    elif dataset == "exchange":
        # return atp_pipeline(num_heads=6, projection_shape_for_head=12, output_shape=32, rate=0.0, permutation_repeats=0,
        #          bound_std=False, num_layers=4, enc_dim=32, xmin=0.1, xmax=1)
        #the above is what i used for previous reults

        # return atp_pipeline(num_heads=10, projection_shape_for_head=9, output_shape=32, rate=0.05, permutation_repeats=0,
        #          bound_std=False, num_layers=6, enc_dim=32, xmin=0.1, xmax=1,MHAX_leakage="new_block")

        return atp_pipeline(num_heads=8, projection_shape_for_head=12, output_shape=32, rate=0.05, permutation_repeats=0,
                 bound_std=False, num_layers=3, enc_dim=32, xmin=0.1, xmax=1,MHAX_leakage="xxx")

    elif dataset == "electricity":
        return atp_pipeline(num_heads=6, projection_shape_for_head=11, output_shape=32, rate=0.25, permutation_repeats=0,
                 bound_std=False, num_layers=4, enc_dim=32, xmin=0.1, xmax=1,MHAX_leakage="xxx")      


    else:
        print('choose a valid dataset name')         
            
        

