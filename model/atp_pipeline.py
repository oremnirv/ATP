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
        self.n_C_s = [120, 20]
        self.n_T_s = [0, 20]
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
        self._DE = DE(self.n_C_s, self.n_T_s, img_seg=self.img_seg) 
    
    # def concat_context_multi_ts(self, list_of_inputs, dim, n_C):
    #     return tf.concat([list_of_inputs[i][dim][:, :n_C[i], :] for i in range(len(list_of_inputs))], axis=1)

    

    
    @tf.function
    def concatenate_tensors(self, tensor, n_C):
        def extract_elements_up_to_index(tensor, indexing_tensor, index_position):
            # Use tf.gather to extract the specific index for each row
            end_index = tf.gather(indexing_tensor, index_position)
            end_index = tf.squeeze(end_index)
            # print("end_index", end_index)
            result = tensor[:, :end_index, :] 
            # print("result", result)
            return result
        # Create a TensorArray
        tensor_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, infer_shape=False)

        n_C_shape = n_C.shape[0]
        # print("n_C_shape", n_C_shape)
        # Write tensors to the TensorArray
        for i in tf.range(n_C_shape):
            r_temp = extract_elements_up_to_index(tensor, n_C, i)
            tensor_array = tensor_array.write(i, r_temp) 

        # print("tensor_array", tensor_array)        
        # Read tensors to a python list, ensuring to keep the Tensor shapes
        tensor_array_stacked = tensor_array.stack()
        # Concatenate the tensors along axis 1
        concatenated_tensor = tf.concat(tensor_array_stacked, axis=1)
        # print("concatenated_tensor", concatenated_tensor)
        return concatenated_tensor


    def concat_context_multi_ts(self, tensor):
        tensor0 = tf.concat([tensor[i][0] for i in range(len(tensor))], axis=1)
        tensor1 = tf.concat([tensor[i][1] for i in range(len(tensor))], axis=1)
        tensor2 = tf.concat([tensor[i][2] for i in range(len(tensor))], axis=1)
        tensor3 = tf.concat([tensor[i][3] for i in range(len(tensor))], axis=1)
        tensor4 = tf.concat([tensor[i][4] for i in range(len(tensor))], axis=1)
        tensor5 = tf.concat([tensor[i][5] for i in range(len(tensor))], axis=1)
        tensor6 = tf.concat([tensor[i][6] for i in range(len(tensor))], axis=1)

        return [tensor0, tensor1, tensor2, tensor3, tensor4, tensor5, tensor6]


    def concat_target_multi_ts(self, list_of_inputs, dim, n_C, n_T, last_dim):
        x = tf.concat([list_of_inputs[i][dim][:, n_C[i]:n_C[i]+n_T[i], :][:, :, tf.newaxis, :] for i in range(len(list_of_inputs))], axis=2)
        x = tf.reshape(x, (x.shape[0], -1, last_dim))
        return x

    def inputs_for_multi_ts(self, x, y, n_C, n_T, n_C_s, n_T_s, labels = None):
        batch_size = x.shape[0]
        
        inputs_for_processing = []
        # print(x.shape)
        eye = tf.eye(33, dtype=tf.float32)
        ## pick the rows of the eye matrix that correspond to the ts
        # print("labels", labels)
        labels = tf.cast(labels, tf.int32)
        ts_labels  = tf.gather(eye, labels, axis=0)
        # print("ts_labels", ts_labels)
        labels_len  = len(labels) 
        ts_start = 0
        # total_length = 180
        for i in range(labels_len):
            print("i", i)
            # embed each ts separately and each dimension separately    
            ts_label = ts_labels[i, :]
            # ts_label = tf.reshape(tf.repeat(ts_label[tf.newaxis, :], batch_size*(total_length), axis=0), (batch_size, -1, labels_l)) # one hot encoding of the ts 
            if (i == (labels_len - 1)):
                total_length = 61
            else:
                total_length = 180
            ts_end = ts_start + (total_length)

            if self._subsample == True:
                # print("subsample")
                # indices = tf.range(ts_start, ts_end)
                x_temp = x[:, ts_start:ts_end, :]
                y_temp = y[:, ts_start:ts_end, :]

                
                if (i == (labels_len - 1)):

                    x_temp, y_temp = self._feature_wrapper.subsample_con_tar(x_temp, y_temp, 60, 120, 20, 10)
                    ts_label = tf.reshape(tf.repeat(ts_label[tf.newaxis, :], batch_size*30, axis=0), (batch_size, -1, 33)) # one hot encoding of the ts
                    x_temp = tf.reshape(x_temp, (batch_size, 30, -1))
                    # print("x_temp.shape OBS:", x_temp.shape)
                    y_temp = tf.reshape(y_temp, (batch_size, 30, -1))
                    # print("here")
                else:   
                    x_temp, y_temp = self._feature_wrapper.subsample_con(x_temp, y_temp, 180, 120)
                    ts_label = tf.reshape(tf.repeat(ts_label[tf.newaxis, :], batch_size*120, axis=0), (batch_size, -1, 33)) # one hot encoding of the ts
                    x_temp = tf.reshape(x_temp, (batch_size, 120, -1))
                    y_temp = tf.reshape(y_temp, (batch_size, 120, -1))


            else:
                x_temp = x[:, ts_start:ts_end, :]
                y_temp = y[:, ts_start:ts_end, :]
                if (i == (labels_len - 1)):

                    ts_label = tf.reshape(tf.repeat(ts_label[tf.newaxis, :], batch_size*61, axis=0), (batch_size, -1, 33)) # one hot encoding of the ts
                    x_temp = tf.reshape(x_temp, (batch_size, 61, -1))
                    y_temp = tf.reshape(y_temp, (batch_size, 61, -1))
                else:   
                    ts_label = tf.reshape(tf.repeat(ts_label[tf.newaxis, :], batch_size*180, axis=0), (batch_size, -1, 33)) # one hot encoding of the ts
                    x_temp = tf.reshape(x_temp, (batch_size, 180, -1))
                    y_temp = tf.reshape(y_temp, (batch_size, 180, -1))


            # print("x_temp.shape", x_temp.shape)
            
            # print("ts_label.shape", ts_label.shape)
            x_emb = [tf.concat([self._feature_wrapper.PE([x_temp[:, :, dim_num][:, :, tf.newaxis], self.enc_dim, self.xmin, self.xmax]), ts_label], axis=-1) for dim_num in range(x_temp.shape[-1])] 
            x_emb = tf.concat(x_emb, axis=-1) # (32, 30, 34)

            # print("x_emb.shape", x_emb.shape) 
            # take derivative of each ts separately
            if self._subsample:
                # print("subsample")

                if (i == (labels_len - 1)):
                    x_temp_context = x_temp[:, :20, :]
                    y_temp_context = y_temp[:, :20, :]
                    x_temp_target = x_temp[:, 20:30, :]
                    y_temp_target = y_temp[:, 20:30, :]

                    y_diff, x_diff, d, x_n, y_n = self._DE([y_temp, x_temp, y_temp_context, y_temp_target, x_temp_context, x_temp_target, 20, 10, i, True]) #  (32, 30, 1),  (32, 30, 1), (32, 30, 2), (32, 30, 1), (32, 30, 1)
                    y_diff = tf.reshape(y_diff, (batch_size, 30, 1))
                    x_diff = tf.reshape(x_diff, (batch_size, 30, 1))
                    d = tf.reshape(d, (batch_size, 30, 2))
                    x_n = tf.reshape(x_n, (batch_size, 30, 1))
                    y_n = tf.reshape(y_n, (batch_size, 30, 1))
   
                else:
                    x_temp_context = x_temp[:, :120, :]
                    y_temp_context = y_temp[:, :120, :]
                    x_temp_target = x_temp[:, 120:120, :]
                    y_temp_target = y_temp[:, 120:120, :]
                    x_temp_context = tf.reshape(x_temp_context, (batch_size, 120, 1))
                    y_temp_context = tf.reshape(y_temp_context, (batch_size, 120, 1))
                    x_temp_target = tf.reshape(x_temp_target, (batch_size, 0, 1))
                    y_temp_target = tf.reshape(y_temp_target, (batch_size, 0, 1))
                    y_diff, x_diff, d, x_n, y_n = self._DE([y_temp, x_temp, y_temp_context, y_temp_target, x_temp_context, x_temp_target, 120, 0, i, True])
                    y_diff = tf.reshape(y_diff, (batch_size, 120, 1))
                    x_diff = tf.reshape(x_diff, (batch_size, 120, 1))
                    d = tf.reshape(d, (batch_size, 120, 2))
                    x_n = tf.reshape(x_n, (batch_size, 120, 1))
                    y_n = tf.reshape(y_n, (batch_size, 120, 1))

            else:
                if (i == (labels_len - 1)):
                    x_temp_context = x_temp[:, :60, :]
                    y_temp_context = y_temp[:, :60, :]
                    x_temp_target = x_temp[:, 60:61, :]
                    y_temp_target = y_temp[:, 60:61, :]

                    y_diff, x_diff, d, x_n, y_n = self._DE([y_temp, x_temp, y_temp_context, y_temp_target, x_temp_context, x_temp_target, 20, 10, i, True]) #  (32, 30, 1),  (32, 30, 1), (32, 30, 2), (32, 30, 1), (32, 30, 1)
                    y_diff = tf.reshape(y_diff, (batch_size, 61, 1))
                    x_diff = tf.reshape(x_diff, (batch_size, 61, 1))
                    d = tf.reshape(d, (batch_size, 61, 2))
                    x_n = tf.reshape(x_n, (batch_size, 61, 1))
                    y_n = tf.reshape(y_n, (batch_size, 61, 1))
   
                else:
                    x_temp_context = x_temp[:, :180, :]
                    y_temp_context = y_temp[:, :180, :]
                    x_temp_target = x_temp[:, 180:180, :]
                    y_temp_target = y_temp[:, 180:180, :]
                    x_temp_context = tf.reshape(x_temp_context, (batch_size, 180, 1))
                    y_temp_context = tf.reshape(y_temp_context, (batch_size, 180, 1))
                    x_temp_target = tf.reshape(x_temp_target, (batch_size, 0, 1))
                    y_temp_target = tf.reshape(y_temp_target, (batch_size, 0, 1))
                    y_diff, x_diff, d, x_n, y_n = self._DE([y_temp, x_temp, y_temp_context, y_temp_target, x_temp_context, x_temp_target, 180, 0, i, True])
                    y_diff = tf.reshape(y_diff, (batch_size, 180, 1))
                    x_diff = tf.reshape(x_diff, (batch_size, 180, 1))
                    d = tf.reshape(d, (batch_size, 180, 2))
                    x_n = tf.reshape(x_n, (batch_size, 180, 1))
                    y_n = tf.reshape(y_n, (batch_size, 180, 1))
            # print("i", i)
            # print("y_diff.shape", y_diff.shape)
            # print('##################')
            inputs_for_processing.append([x_emb, y_temp, y_diff, x_diff, d, x_n, y_n])  
            ts_start = ts_end
        

        if self._bc:
            # the end sequence will be (y_11,.., y_1n_C, y21, ..y_2n_C2, y1*, y1**, ...,y1****)    
            context_list = self.concat_context_multi_ts(inputs_for_processing) 
            print("context_list", context_list)
            x_emb, y, y_diff, x_diff, d, x_n, y_n  = context_list
        else:
            # the end sequence will be (y_11,.., y_1n_C, y21, ..y_2n_C, y1*, y2*, ..yk*, y1**, y2**, ..yk**)
            context_list = [self.concat_context_multi_ts(inputs_for_processing, j, n_C) for j in range(7)]
            target_list = [self.concat_target_multi_ts(inputs_for_processing, j, n_C, n_T, context_list[j].shape[-1]) for j in range(7)]
            x_emb, y, y_diff, x_diff, d, x_n, y_n = [tf.concat([context_list[j], target_list[j]], axis=1) for j in range(7)]
        inputs_for_processing = [x_emb, y, y_diff, x_diff, d, x_n, y_n]
        return inputs_for_processing, y, y_n





    def call(self, inputs):

        x, y, n_C, n_T, training, n_C_s, n_T_s, labels = inputs #  (batch_size, n_C + n_T, 1), (batch_size, n_C + n_T, 1)
        labels_l  = len(labels) - 1
        # print("x shape:", x.shape)
        if not self._bc:
            total_length = sum(n_C) + sum(n_T)
            x = x[:,:total_length,:]
            y = y[:,:total_length,:]
        # print("x shape:", x.shape)
        if (self._subsample == True) and (self.multiply == 1):
            x, y = self._feature_wrapper.subsample(x, y, n_C[0], n_T[0], n_C_s[0], n_T_s[0])
            n_C = n_C_s
            n_T = n_T_s
            # print("y shape after subsample:", y.shape)
        #x and y have shape batch size x length x dim
        # if training == True:    
            # x,y = self._feature_wrapper.permute([x, y, n_C * self.multiply, n_T * self.multiply, self._permutation_repeats]) ##### clean permute, and check permute target and/or context?
        # print("y shape after permute:", y.shape)
        # permute returns y with shape batch size x NONE x dim ****** check this issue! ******
        

        if self.multiply == 1:
            x_emb = [self._feature_wrapper.PE([x[:, :, dim_num][:, :, tf.newaxis], self.enc_dim, self.xmin, self.xmax]) for dim_num in range(x.shape[-1])] 
            x_emb = tf.concat(x_emb, axis=-1)  #(32, 30, 32) (batch_size, n_C + n_T, enc_dim) 
            n_C = n_C[0]
            n_T = n_T[0]
            ######## create derivative ########
            y_diff, x_diff, d, x_n, y_n = self._DE([y, x, n_C, n_T, training]) #  (32, 30, 1), (32, 30, 1), (32, 30, 2), (32, 30, 1), (32, 30, 1) [n_C = 20 and n_T = 10]
            inputs_for_processing = [x_emb, y, y_diff, x_diff, d, x_n, y_n]

        else:  
            # # the end sequence will be (y_11,.., y_1n_C, y21, ..y_2n_C, y1*, y2*, ..yk*, y1**, y2**, ..yk**)
            inputs_for_processing, y, y_n = self.inputs_for_multi_ts(x, y, n_C, n_T, n_C_s, n_T_s, labels)
            # print("y shape after inputs_for_processing:", y.shape)
            # print("y shape after inputs_for_processing:", y.shape)
            if (self._subsample):
                pass
        n_C1 = (labels_l * 120) + 20 
        print("n_C1", n_C1)
                  
        inputs_for_processing.append(n_C1)
        inputs_for_processing.append(10)
        query_x, key_x, value_x, query_xy, key_xy, value_xy = self._feature_wrapper(inputs_for_processing) #  (batch_size, multiply *(n_C_s + n_T_s), enc_dim + multiply + x_diff.dim + x_n.dim), (batch_size, multiply *(n_C_s + n_T_s), enc_dim + multiply + x_diff.dim + x_n.dim) , (batch_size, multiply *(n_C_s + n_T_s), 1), (batch_size, multiply *(n_C_s + n_T_s), enc.dim + multiply + 2 + label.dim + y.dim + y_diff.dim + d.dim + y_n.dim)

        value_x = tf.reshape(value_x, (value_x.shape[0], tf.shape(key_x)[1], value_x.shape[-1]))
        y_n_closest = y_n[:, :, :y.shape[-1]] #### need to update this based on how we pick closest point
        ######## make mask #######
        mask = self._feature_wrapper.masker(n_C1, 10)
        μ, log_σ = self._atp([query_x, key_x, value_x, query_xy, key_xy, value_xy, mask, y_n_closest], training=training)
        
        return μ[:, n_C1:], log_σ[:, n_C1:], y[:, n_C1:]
      

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


