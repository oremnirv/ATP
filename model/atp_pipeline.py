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
                 bound_std=False, num_layers=3, enc_dim=32, xmin=0.1, xmax=2, multiply=1, MHAX_leakage=True,**kwargs):
        super().__init__(**kwargs)
        # for testing set permutation_repeats=0
   
        self._permutation_repeats = permutation_repeats
        self.enc_dim = enc_dim
        self.xmin = xmin
        self.xmax = xmax
        self.multiply = multiply
        self._feature_wrapper = feature_wrapper(multiply=self.multiply)
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
        
        x = x[:,:(n_C+n_T) * self.multiply,:]
        y = y[:,:(n_C+n_T) * self.multiply,:]
        # if training == True:    
            # x,y = self._feature_wrapper.permute([x, y, n_C * self.multiply, n_T * self.multiply, self._permutation_repeats]) ##### clean permute, and check permute target and/or context?
        # print("y shape after permute:", y.shape)
        # permute returns y with shape batch size x NONE x dim ****** check this issue! ******
        ######## make mask #######
        
        context_part = tf.concat([tf.ones((n_C * self.multiply, n_C* self.multiply), tf.bool),tf.zeros((n_C * self.multiply, n_T* self.multiply),tf.bool)],axis=-1)
        diagonal_mask = tf.linalg.band_part(tf.ones(((n_C+n_T)* self.multiply, (n_C+n_T)* self.multiply),tf.bool),-1,0)
        lower_diagonal_mask = tf.linalg.set_diag(diagonal_mask,tf.zeros(diagonal_mask.shape[0:-1],tf.bool)) ### condense into one line?                                                                               
        mask = tf.concat([context_part,lower_diagonal_mask[n_C* self.multiply:(n_C+n_T)* self.multiply,:(n_C+n_T)* self.multiply]],axis=0) # check no conflicts with init and check mask is correct shape
        # print(mask.shape)

        if self.multiply == 1:
            x_emb = [self._feature_wrapper.PE([x[:, :, i][:, :, tf.newaxis], self.enc_dim, self.xmin, self.xmax]) for i in range(x.shape[-1])] 
            x_emb = tf.concat(x_emb, axis=-1)

            ######## create derivative ########

            y_diff, x_diff, d, x_n, y_n = self._DE([y, x, n_C, n_T, training])

            inputs_for_processing = [x_emb, y, y_diff, x_diff, d, x_n, y_n, n_C, n_T]

        else: 
            batch_size = x.shape[0]

            inputs_for_processing = []
            eye = tf.eye(self.multiply)
            for i in range(self.multiply):
                x_emb = [tf.concat([self._feature_wrapper.PE([x[:, i*(n_C + n_T):(i+1)*(n_C + n_T), j][:, :, tf.newaxis], 20, 0.1, 2]), tf.reshape(tf.repeat(eye[:, i][tf.newaxis, :], batch_size*(n_C + n_T), axis=0), (batch_size, -1, self.multiply))], axis=-1) for j in range(x.shape[-1])] 
                x_emb = tf.concat(x_emb, axis=-1)
                y_diff, x_diff, d, x_n, y_n = self._DE([y[:, i*(n_C + n_T):(i+1)*(n_C + n_T)], x[:, i*(n_C + n_T):(i+1)*(n_C + n_T), 0][:, :, np.newaxis], n_C, n_T, True])
                inputs_for_processing.append([x_emb, y[:, i*(n_C + n_T):(i+1)*(n_C + n_T)][:, :, np.newaxis], y_diff, x_diff, d, x_n, y_n, n_C, n_T])  

            x_emb_a = tf.concat([inputs_for_processing[i][0][:, :n_C, :] for i in range(len(inputs_for_processing))], axis=1)
            x_emb_b = tf.concat([inputs_for_processing[i][0][:, n_C:n_C+n_T, :][:, :, tf.newaxis, :] for i in range(len(inputs_for_processing))], axis=2)
            x_emb_b = tf.reshape(x_emb_b, (x_emb_b.shape[0], -1, x_emb_a.shape[-1]))
            x_emb_c = tf.concat([x_emb_a, x_emb_b], axis=1)
            # print("x inputs_for_processing 0: ", inputs_for_processing[0][0][:, :n_C, :].shape)
            # print("x inputs_for_processing 1: ", inputs_for_processing[1][0][:, :n_C, :].shape)
            # print("x inputs_for_processing 3: ", inputs_for_processing[3][0][:, :n_C, :].shape)    
            # print("x inputs_for_processing 2: ", inputs_for_processing[2][0][:, :n_C, :].shape)
            # print("x inputs_for_processing 4: ", inputs_for_processing[4][0][:, :n_C, :].shape)
            # print("x inputs_for_processing 5: ", inputs_for_processing[5][0][:, :n_C, :].shape)
            # print("x inputs_for_processing 6: ", inputs_for_processing[6][0][:, :n_C, :].shape)
            # print("x inputs_for_processing 7: ", inputs_for_processing[7][0][:, :n_C, :].shape)            

            zz1 = tf.concat([inputs_for_processing[i][1][:, :n_C, :, 0] for i in range(len(inputs_for_processing))], axis=1)
            zz2 = tf.concat([inputs_for_processing[i][1][:, n_C:n_C+n_T, :, 0] for i in range(len(inputs_for_processing))], axis=2)
            zz2 = tf.reshape(zz2, (zz2.shape[0], -1, 1))
            zz3 = tf.concat([zz1, zz2], axis=1)


            y_diff_a = tf.concat([inputs_for_processing[i][2][:, :n_C, :] for i in range(len(inputs_for_processing))], axis=1)
            y_diff_b = tf.concat([inputs_for_processing[i][2][:, n_C:n_C+n_T, :] for i in range(len(inputs_for_processing))], axis=2)
            y_diff_b = tf.reshape(y_diff_b, (y_diff_b.shape[0], -1, y_diff_a.shape[-1]))
            y_diff_c = tf.concat([y_diff_a, y_diff_b], axis=1)
            # print("y_diff inputs_for_processing 0: ", inputs_for_processing[0][2][:, :n_C, :].shape)

            x_diff_a = tf.concat([inputs_for_processing[i][3][:, :n_C, :] for i in range(len(inputs_for_processing))], axis=1)
            x_diff_b = tf.concat([inputs_for_processing[i][3][:, n_C:n_C+n_T, :] for i in range(len(inputs_for_processing))], axis=2)
            x_diff_b = tf.reshape(x_diff_b, (x_diff_b.shape[0], -1, x_diff_a.shape[-1]))
            x_diff_c = tf.concat([x_diff_a, x_diff_b], axis=1)
            # print("x_diff inputs_for_processing 0: ", inputs_for_processing[0][3][:, :n_C, :].shape)

            d_a = tf.concat([inputs_for_processing[i][4][:, :n_C, :] for i in range(len(inputs_for_processing))], axis=1)
            d_b = tf.concat([inputs_for_processing[i][4][:, n_C:n_C+n_T, :][:, :, tf.newaxis, :] for i in range(len(inputs_for_processing))], axis=2)
            d_b = tf.reshape(d_b, (d_b.shape[0], -1, d_a.shape[-1]))
            d_c = tf.concat([d_a, d_b], axis=1)
            # print("d inputs_for_processing 0: ", inputs_for_processing[0][4][:, :n_C, :].shape)

            x_n_a = tf.concat([inputs_for_processing[i][5][:, :n_C, :] for i in range(len(inputs_for_processing))], axis=1)
            x_n_b = tf.concat([inputs_for_processing[i][5][:, n_C:n_C+n_T, :] for i in range(len(inputs_for_processing))], axis=2)
            x_n_b = tf.reshape(x_n_b, (x_n_b.shape[0], -1, x_n_a.shape[-1]))
            x_n_c = tf.concat([x_n_a, x_n_b], axis=1)
            # print("x_n inputs_for_processing 0: ", inputs_for_processing[0][5][:, :n_C, :].shape)

            y_n_a = tf.concat([inputs_for_processing[i][6][:, :n_C, :] for i in range(len(inputs_for_processing))], axis=1)
            y_n_b = tf.concat([inputs_for_processing[i][6][:, n_C:n_C+n_T, :] for i in range(len(inputs_for_processing))], axis=2)
            y_n_b = tf.reshape(y_n_b, (y_n_b.shape[0], -1, y_n_a.shape[-1]))
            y_n_c = tf.concat([y_n_a, y_n_b], axis=1)
            # print("y_n inputs_for_processing 0: ", inputs_for_processing[0][6][:, :n_C, :].shape)
            y_n = y_n_c
            inputs_for_processing = [x_emb_c, zz3, y_diff_c, x_diff_c, d_c, x_n_c, y_n_c, n_C , n_T]
        # print("after processing")
        # print(x_emb_c.shape, zz3.shape, y_diff_c.shape, x_diff_c.shape, d_c.shape, x_n_c.shape, y_n_c.shape)
        # print("#####")
        query_x, key_x, value_x, query_xy, key_xy, value_xy = self._feature_wrapper(inputs_for_processing)
        
        y_n_closest = y_n[:, :, :y.shape[-1]] #### need to update this based on how we pick closest point

        μ, log_σ = self._atp([query_x, key_x, value_x, query_xy, key_xy, value_xy, mask, y_n_closest],training=training)

        return μ[:, n_C*self.multiply:], log_σ[:, n_C*self.multiply:]
      

def instantiate_atp(dataset,training=True):
    if dataset == "ETT":
        return atp_pipeline(num_heads=4, projection_shape_for_head=4, output_shape=64, rate=0.1, permutation_repeats=1,
                    bound_std=False, num_layers=2, enc_dim=32, xmin=0.1, xmax=2)
    elif dataset == "traffic":
        return atp_pipeline(num_heads=4, projection_shape_for_head=4, output_shape=64, rate=0.1, permutation_repeats=1,
                    bound_std=False, num_layers=2, enc_dim=32, xmin=0.1, xmax=2)
    elif dataset == "exchange":
        # return atp_pipeline(num_heads=6, projection_shape_for_head=12, output_shape=32, rate=0.0, permutation_repeats=0,
        #          bound_std=False, num_layers=4, enc_dim=32, xmin=0.1, xmax=1)
        #the above is what i used for previous reults

        return atp_pipeline(num_heads=10, projection_shape_for_head=9, output_shape=32, rate=0.05, permutation_repeats=0,
                 bound_std=False, num_layers=6, enc_dim=32, xmin=0.1, xmax=1,MHAX_leakage="new_block")

    else:
        print('choose a valid dataset name')         
            
        

