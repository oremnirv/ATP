import tensorflow as tf
from tensorflow import keras
import numpy as np
from data_wrangler import batcher, dataset_preparer
from data_wrangler.feature_extractor import  DE, feature_wrapper
from model.atp import ATP
from model.atp_no_leakage import ATP as ATP_no_leakage
from model.atp_no_leakage_new_block import ATP as ATP_new_block
from model.atp_no_leakage_xxx import ATP as ATP_no_leakage_xxx



class atp_pipeline(keras.models.Model):
    
    def __init__(self, num_heads=4, projection_shape_for_head=4, output_shape=64, rate=0.1, permutation_repeats=1,
                 bound_std=False, num_layers=3, enc_dim=32, xmin=0.1, xmax=2, multiply=1, MHAX_leakage=True, subsample =True, bc = False, y_target_dim=1, img_seg=False):
        super().__init__()
        # for testing set permutation_repeats=0
        self._MHAX_leakage = MHAX_leakage
        self._permutation_repeats = permutation_repeats
        self.enc_dim = enc_dim
        self.img_seg = img_seg
        self.xmin = xmin
        self.xmax = xmax
        self.y_target_dim = y_target_dim
        self.multiply = multiply
        self._subsample = subsample
        self._bc = bc   
        self._feature_wrapper = feature_wrapper()
        if self._MHAX_leakage == True:
            self._atp = ATP(num_heads=num_heads,dropout_rate=rate,num_layers=num_layers,output_shape=output_shape,
                        projection_shape=projection_shape_for_head*num_heads,bound_std=bound_std, y_target_dim=self.y_target_dim)
        elif self._MHAX_leakage == False:
            self._atp = ATP_no_leakage(num_heads=num_heads,dropout_rate=rate,num_layers=num_layers,output_shape=output_shape,
                        projection_shape=projection_shape_for_head*num_heads,bound_std=bound_std, y_target_dim=self.y_target_dim)
        elif self._MHAX_leakage == "new_block":
            self._atp = ATP_new_block(num_heads=num_heads,dropout_rate=rate,num_layers=num_layers,output_shape=output_shape,
                        projection_shape=projection_shape_for_head*num_heads,bound_std=bound_std, y_target_dim=self.y_target_dim)
        elif self._MHAX_leakage == "xxx":
            self._atp = ATP_no_leakage_xxx(num_heads=num_heads,dropout_rate=rate,num_layers=num_layers,output_shape=output_shape,
                        projection_shape=projection_shape_for_head*num_heads,bound_std=bound_std, y_target_dim=self.y_target_dim)
        self._DE = DE(img_seg=self.img_seg) 
    
    def inputs_for_multi_ts(self, x, y, labels = None):
        batch_size = x.shape[0]

        def loop_body(i, ts_start, inputs_for_processing1, inputs_for_processing2, inputs_for_processing3, inputs_for_processing4, inputs_for_processing5, inputs_for_processing6, inputs_for_processing7, ts_labels, labels_len):
            # print('i: ', i)
            ts_label1 = ts_labels[i, :]

        # Example conditional logic
            total_length = tf.cond(
            tf.equal(i, labels_len - 1),
            lambda: tf.constant(180 if self._subsample else 61),  # True case
            lambda: tf.constant(180)  # False case
            )
    
            # Example tensor manipulation (simplified)
            ts_end = ts_start + total_length
            x_temp = x[:, ts_start:ts_end, :]
            y_temp = y[:, ts_start:ts_end, :]

            if self._subsample == True:

                x_temp, y_temp  = tf.cond(
                    tf.equal(i, labels_len - 1),
                    lambda: self._feature_wrapper.subsample_con_tar(x_temp, y_temp, 60, 120, 10, 5),  # True case
                    lambda: self._feature_wrapper.subsample_con(x_temp, y_temp, 180, 120)  # False case
                )
                ts_label1 = tf.cond(
                    tf.equal(i, labels_len - 1),
                    lambda: tf.reshape(tf.repeat(ts_label1[tf.newaxis, :], batch_size * 15, axis=0), (batch_size, -1, 33)),
                    lambda: tf.reshape(tf.repeat(ts_label1[tf.newaxis, :], batch_size*120, axis=0), (batch_size, -1, 33)) # one hot encoding of the ts
                )
                x_temp_context, y_temp_context, x_temp_target, y_temp_target = tf.cond(
                    tf.equal(i, labels_len - 1),
                    lambda: (x_temp[:, :10, :], y_temp[:, :10, :], x_temp[:, 10:15, :], y_temp[:, 10:15, :]),  # True case
                    lambda: (x_temp[:, :120, :], y_temp[:, :120, :], x_temp[:, 120:120, :], y_temp[:, 120:120, :])  # False case
                )
                y_diff, x_diff, d, x_n, y_n  = tf.cond(
                    tf.equal(i, labels_len - 1),
                    lambda: self._DE([y_temp, x_temp, y_temp_context, y_temp_target, x_temp_context, x_temp_target, 10, 5, i, True]),  # True case
                    lambda: self._DE([y_temp, x_temp, y_temp_context, y_temp_target, x_temp_context, x_temp_target, 120, 0, i, True])  # False case
                )
                  # No padding for dimensions 0 and 2


            else:

                ts_label1 = tf.cond(
                    tf.equal(i, labels_len - 1),
                    lambda: tf.reshape(tf.repeat(ts_label1[tf.newaxis, :], batch_size * 61, axis=0), (batch_size, -1, 33)),
                    lambda: tf.reshape(tf.repeat(ts_label1[tf.newaxis, :], batch_size*180, axis=0), (batch_size, -1, 33)) # one hot encoding of the ts
                )

                x_temp_context, y_temp_context, x_temp_target, y_temp_target = tf.cond(
                    tf.equal(i, labels_len - 1),
                    lambda: (x_temp[:, :60, :], y_temp[:, :60, :], x_temp[:, 60:61, :], y_temp[:, 60:61, :]),  # True case
                    lambda: (x_temp[:, :180, :], y_temp[:, :180, :], x_temp[:, 180:180, :], y_temp[:, 180:180, :])  # False case
                )
                y_diff, x_diff, d, x_n, y_n  = tf.cond(
                    tf.equal(i, labels_len - 1),
                    lambda: self._DE([y_temp, x_temp, y_temp_context, y_temp_target, x_temp_context, x_temp_target, 60, 1, i, True]),  # True case
                    lambda: self._DE([y_temp, x_temp, y_temp_context, y_temp_target, x_temp_context, x_temp_target, 180, 0, i, True])  # False case
                )

                # paddings = [[0, 0], [0, 180 - 61], [0, 0]]  # No padding for dimensions 0 and 2

            paddings = [[0, 0], [0, 120 - 15], [0, 0]]
            x_temp = tf.cast(x_temp, tf.float32)
            x_emb = [tf.concat([self._feature_wrapper.PE([x_temp[:, :, dim_num][:, :, tf.newaxis], self.enc_dim, self.xmin, self.xmax]), ts_label1], axis=-1) for dim_num in range(x_temp.shape[-1])] 
            x_emb = tf.concat(x_emb, axis=-1)
            y_temp = tf.cast(y_temp, tf.float32)
            x_emb = tf.cond(
                tf.equal(i, labels_len - 1),
                lambda: tf.pad(x_emb, paddings, "CONSTANT"),  # True case
                lambda: x_emb  # False case
            )
            

            y_temp = tf.cond(tf.equal(i, labels_len - 1),
                    lambda: tf.pad(y_temp, paddings, "CONSTANT"),  # True case
                    lambda: y_temp)
            y_diff = tf.cond(
                tf.equal(i, labels_len - 1),
                lambda: tf.pad(y_diff, paddings, "CONSTANT"),  # True case
                lambda: y_diff  # False case
            )
            x_diff = tf.cond(
                tf.equal(i, labels_len - 1),
                lambda: tf.pad(x_diff, paddings, "CONSTANT"),  # True case
                lambda: x_diff  # False case
            )
            d = tf.cond(
                tf.equal(i, labels_len - 1),
                lambda: tf.pad(d, paddings, "CONSTANT"),  # True case
                lambda: d  # False case
            )
            x_n = tf.cond(
                tf.equal(i, labels_len - 1),
                lambda: tf.pad(x_n, paddings, "CONSTANT"),  # True case
                lambda: x_n  # False case
            )
            y_n = tf.cond(
                tf.equal(i, labels_len - 1),
                lambda: tf.pad(y_n, paddings, "CONSTANT"),  # True case
                lambda: y_n  # False case
            )
            # print("y_n shape", y_n.shape)

            inputs_for_processing_new1 = inputs_for_processing1.write(i, x_emb)
            inputs_for_processing_new2 = inputs_for_processing2.write(i, y_temp)
            inputs_for_processing_new3 = inputs_for_processing3.write(i, y_diff)
            inputs_for_processing_new4 = inputs_for_processing4.write(i, x_diff)
            inputs_for_processing_new5 = inputs_for_processing5.write(i, d)
            inputs_for_processing_new6 = inputs_for_processing6.write(i, x_n) 
            inputs_for_processing_new7 = inputs_for_processing7.write(i, y_n) 

            
        
            return (i+1, ts_end, inputs_for_processing_new1, inputs_for_processing_new2, inputs_for_processing_new3, inputs_for_processing_new4, inputs_for_processing_new5, inputs_for_processing_new6, inputs_for_processing_new7, ts_labels, labels_len)
        
        def condition(i, ts_start, inputs_for_processing1, inputs_for_processing2, inputs_for_processing3, inputs_for_processing4, inputs_for_processing5, inputs_for_processing6, inputs_for_processing7, ts_labels, labels_len):
            return i < labels_len

        i = tf.constant(0)
        ts_start = tf.constant(0)
        inputs_for_processing1 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        inputs_for_processing2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        inputs_for_processing3 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        inputs_for_processing4 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        inputs_for_processing5 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        inputs_for_processing6 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        inputs_for_processing7 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        labels_len = tf.shape(y)[1] // 180
        eye = tf.eye(33, dtype=tf.float32)
        ts_labels  = tf.gather(eye, labels, axis=0)
        i, ts_start, inputs_for_processing1, inputs_for_processing2, inputs_for_processing3, inputs_for_processing4, inputs_for_processing5, inputs_for_processing6, inputs_for_processing7, ts_labels, labels_len  = tf.while_loop(condition, loop_body, [i, ts_start, inputs_for_processing1, inputs_for_processing2, inputs_for_processing3, inputs_for_processing4, inputs_for_processing5, inputs_for_processing6, inputs_for_processing7, ts_labels, labels_len])
        inputs_for_processing1 = inputs_for_processing1.stack()
        inputs_for_processing2 = inputs_for_processing2.stack()
        inputs_for_processing3 = inputs_for_processing3.stack()
        inputs_for_processing4 = inputs_for_processing4.stack()
        inputs_for_processing5 = inputs_for_processing5.stack()
        inputs_for_processing6 = inputs_for_processing6.stack()
        inputs_for_processing7 = inputs_for_processing7.stack()
        inputs_for_processing1 = tf.transpose(inputs_for_processing1, [1, 0, 2, 3])
        inputs_for_processing2 = tf.transpose(inputs_for_processing2, [1, 0, 2, 3])
        inputs_for_processing3 = tf.transpose(inputs_for_processing3, [1, 0, 2, 3])
        inputs_for_processing4 = tf.transpose(inputs_for_processing4, [1, 0, 2, 3])
        inputs_for_processing5 = tf.transpose(inputs_for_processing5, [1, 0, 2, 3])
        inputs_for_processing6 = tf.transpose(inputs_for_processing6, [1, 0, 2, 3])
        inputs_for_processing7 = tf.transpose(inputs_for_processing7, [1, 0, 2, 3])
        inputs_for_processing1 = tf.reshape(inputs_for_processing1, [batch_size, -1, inputs_for_processing1.shape[-1]])
        inputs_for_processing2 = tf.reshape(inputs_for_processing2, [batch_size, -1, inputs_for_processing2.shape[-1]])
        inputs_for_processing3 = tf.reshape(inputs_for_processing3, [batch_size, -1, inputs_for_processing3.shape[-1]])
        inputs_for_processing4 = tf.reshape(inputs_for_processing4, [batch_size, -1, inputs_for_processing4.shape[-1]])
        inputs_for_processing5 = tf.reshape(inputs_for_processing5, [batch_size, -1, inputs_for_processing5.shape[-1]])
        inputs_for_processing6 = tf.reshape(inputs_for_processing6, [batch_size, -1, inputs_for_processing6.shape[-1]])
        inputs_for_processing7 = tf.reshape(inputs_for_processing7, [batch_size, -1, inputs_for_processing7.shape[-1]])
        # print("inputs_for_processing7 shape", inputs_for_processing7.shape)

        return inputs_for_processing1, inputs_for_processing2, inputs_for_processing3, inputs_for_processing4, inputs_for_processing5, inputs_for_processing6, inputs_for_processing7

    def call(self, inputs):

        x, y, _, _, training, _, _, labels = inputs #  (batch_size, n_C + n_T, 1), (batch_size, n_C + n_T, 1)
        labels_l  = len(labels) - 1
        inputs_for_processing1, inputs_for_processing2, inputs_for_processing3, inputs_for_processing4, inputs_for_processing5, inputs_for_processing6, inputs_for_processing7 = self.inputs_for_multi_ts(x, y, labels)
        inputs_for_processing = [inputs_for_processing1, inputs_for_processing2, inputs_for_processing3, inputs_for_processing4, inputs_for_processing5, inputs_for_processing6, inputs_for_processing7]


        if self._subsample:
            n_C1 = (labels_l * 120) + 10 
        else:
            n_C1 = (labels_l * 180) + 60
                  
        
        inputs_for_processing.append(n_C1)
        if self._subsample:
            inputs_for_processing.append(110)
        else:
            inputs_for_processing.append(120)
        # print("inputs_for_processing shape", inputs_for_processing[-1])
        query_x, key_x, value_x, query_xy, key_xy, value_xy = self._feature_wrapper(inputs_for_processing) #  (batch_size, multiply *(n_C_s + n_T_s), enc_dim + multiply + x_diff.dim + x_n.dim), (batch_size, multiply *(n_C_s + n_T_s), enc_dim + multiply + x_diff.dim + x_n.dim) , (batch_size, multiply *(n_C_s + n_T_s), 1), (batch_size, multiply *(n_C_s + n_T_s), enc.dim + multiply + 2 + label.dim + y.dim + y_diff.dim + d.dim + y_n.dim)
        value_x = tf.reshape(value_x, (value_x.shape[0], tf.shape(key_x)[1], value_x.shape[-1]))
        y_n_closest = inputs_for_processing[-3]#[:, :, :inputs_for_processing[1].shape[-1]] 
        ######## make mask #######
        if self._subsample:
            mask = self._feature_wrapper.masker(n_C1, 110)
        else:
            mask = self._feature_wrapper.masker(n_C1, 120)
        # print("mask shape", mask.shape)
        μ, log_σ = self._atp([query_x, key_x, value_x, query_xy, key_xy, value_xy, mask, y_n_closest], training=training)
        # print("μ shape", μ.shape)
        # print("log_σ shape", log_σ.shape)
        return μ[:, n_C1:], log_σ[:, n_C1:], inputs_for_processing[1][:, n_C1:]
    #  
      

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
            
        

if __name__ == 'main':
    x_train, y_train, x_val, y_val, x_test, y_test = dataset_preparer.dataset_processor(path_to_data="datasets/ETTm2.csv") 
    multiply = 2
    n_C_s, n_T_s = 20, 10
    n_C, n_T = 96, 192
    idx_list = list(np.arange(0, x_train.shape[0] - 288, 1))
    x, y, _, _ =batcher.batcher(x_train, y_train, idx_list=idx_list, window=n_C + n_T, batch_s=1)
    ### test the one hot encoding of the ts, change multiply (above) to test different ts labels
    for i in range(multiply):
        batch_size = x.shape[0]
        eye = tf.eye(multiply)
        ts_label = tf.reshape(tf.repeat(eye[:, i][tf.newaxis, :], batch_size*(n_C + n_T), axis=0), (batch_size, -1, multiply)) # one hot encoding of the ts
        print("ts_label shape", ts_label.shape)
        print("ts_label example", ts_label[0, 0, :])
    #####


