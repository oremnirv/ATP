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
            print("result", result)
            return result
        # Create a TensorArray
        tensor_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, infer_shape=False)

        n_C_shape = n_C.shape[0]
        # print("n_C_shape", n_C_shape)
        # Write tensors to the TensorArray
        for i in tf.range(n_C_shape):
            r_temp = extract_elements_up_to_index(tensor, n_C, i)
            tensor_array = tensor_array.write(i, r_temp) 

        print("tensor_array", tensor_array)        
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
        print("tensor0.shape", tensor0.shape)
        print("tensor1.shape", tensor1.shape)
        print("tensor2.shape", tensor2.shape)
        print("tensor3.shape", tensor3.shape)
        print("tensor4.shape", tensor4.shape)
        print("tensor5.shape", tensor5.shape)
        print("tensor6.shape", tensor6.shape)
        # n_C1 = tf.constant([120, 20])
        # a1  = self.concatenate_tensors(tensor0, n_C1)
        # a1 = tf.reshape(a1, (a1.shape[0], -1, a1.shape[-1]))
        # print("a1.shape", a1.shape)
        a1 = tf.expand_dims(tensor0, axis=3)
        a2 = tf.expand_dims(tensor1, axis=3)
        a3 = tf.expand_dims(tensor2, axis=3)
        a4 = tf.expand_dims(tensor3, axis=3)
        a5 = tf.expand_dims(tensor4, axis=3)
        a6 = tf.expand_dims(tensor5, axis=3)
        a7 = tf.expand_dims(tensor6, axis=3)
        # a2 = tf.expand_dims(self.concatenate_tensors(tensor1, n_C1), axis=3)
        # a3 = tf.expand_dims(self.concatenate_tensors(tensor2, n_C1), axis=3)
        # a4 = tf.expand_dims(self.concatenate_tensors(tensor3, n_C1), axis=3)
        # a5 = tf.expand_dims(self.concatenate_tensors(tensor4, n_C1), axis=3)
        # a6 = tf.expand_dims(self.concatenate_tensors(tensor5, n_C1), axis=3)
        # a7 = tf.expand_dims(self.concatenate_tensors(tensor6, n_C1), axis=3)

        return [a1, a2, a3, a4, a5, a6, a7]


    def concat_target_multi_ts(self, list_of_inputs, dim, n_C, n_T, last_dim):
        x = tf.concat([list_of_inputs[i][dim][:, n_C[i]:n_C[i]+n_T[i], :][:, :, tf.newaxis, :] for i in range(len(list_of_inputs))], axis=2)
        x = tf.reshape(x, (x.shape[0], -1, last_dim))
        return x

    def inputs_for_multi_ts(self, x, y, n_C, n_T, n_C_s, n_T_s):
        idx_c_all = []
        idx_t_all = []
        batch_size = x.shape[0]
        inputs_for_processing = []
        eye = tf.eye(self.multiply)
        ts_start = 0
        for i in range(self.multiply):
            
            # embed each ts separately and each dimension separately
            total_length = tf.cast(n_C[i] + n_T[i], tf.int32)
            ts_label = tf.reshape(tf.repeat(eye[:, i][tf.newaxis, :], batch_size*(total_length), axis=0), (batch_size, -1, self.multiply)) # one hot encoding of the ts
            ts_end = ts_start + (total_length)
            tf.print("ts_end", ts_end)
            # print("ts_start", ts_start)

            if self._subsample == True:
                print("subsample")
                x_temp, y_temp, idx_c, idx_t = self._feature_wrapper.subsample(x[:, ts_start:ts_end, :], y[:, ts_start:ts_end, :], n_C[i], n_T[i], n_C_s[i], n_T_s[i])

                ts_label = tf.reshape(tf.repeat(eye[:, i][tf.newaxis, :], batch_size*(n_C_s[i] + n_T_s[i]), axis=0), (batch_size, -1, self.multiply)) # overwrites the previous ts_label
                idx_c_all.append(idx_c)
                idx_t_all.append(idx_t)
                # print("finished subsample")
            else:
                x_temp = x[:, ts_start:ts_end, :]
                y_temp = y[:, ts_start:ts_end, :]

            # print("x_temp.shape", x_temp.shape)
            x_emb = [tf.concat([self._feature_wrapper.PE([x_temp[:, :, dim_num][:, :, tf.newaxis], self.enc_dim, self.xmin, self.xmax]), ts_label], axis=-1) for dim_num in range(x_temp.shape[-1])] 
            x_emb = tf.concat(x_emb, axis=-1) # (32, 30, 34)
            # print("x_emb.shape", x_emb.shape) 
            # take derivative of each ts separately
            if self._subsample:
                print("subsample")
                x_temp_context = x_temp[:, :n_C_s[i], :]
                y_temp_context = y_temp[:, :n_C_s[i], :]
                x_temp_target = x_temp[:, n_C_s[i]:n_C_s[i]+n_T_s[i], :]
                y_temp_target = y_temp[:, n_C_s[i]:n_C_s[i]+n_T_s[i], :]

                if i == 0:
                    print("i==0")
                    y_diff, x_diff, d, x_n, y_n = self._DE([y_temp_context, y_temp_target, x_temp_context, x_temp_target, 120, 0, i, True]) #  (32, 30, 1),  (32, 30, 1), (32, 30, 2), (32, 30, 1), (32, 30, 1)
                    print("y_diff.shape", y_diff.shape)
                    print("x_diff.shape", x_diff.shape)
                    print("d.shape", d.shape)
                    print("x_n.shape", x_n.shape)
                    print("y_n.shape", y_n.shape)
   
                else:
                    print("i!=0")
                    y_diff, x_diff, d, x_n, y_n = self._DE([y_temp_context, y_temp_target, x_temp_context, x_temp_target, 20, 20, i, True])
                    y_diff = tf.reshape(y_diff, (batch_size, n_C_s[i]+n_T_s[i], 1))
                    x_diff = tf.reshape(x_diff, (batch_size, n_C_s[i]+n_T_s[i], 1))
                    d = tf.reshape(d, (batch_size, n_C_s[i]+n_T_s[i], 2))
                    x_n = tf.reshape(x_n, (batch_size, n_C_s[i]+n_T_s[i], 1))
                    y_n = tf.reshape(y_n, (batch_size, n_C_s[i]+n_T_s[i], 1))

            else:
                y_diff, x_diff, d, x_n, y_n = self._DE([y_temp, x_temp, n_C[i], n_T[i], True]) #  (32, 30, 1),  (32, 30, 1), (32, 30, 2), (32, 30, 1), (32, 30, 1)
            
            inputs_for_processing.append([x_emb, y_temp, y_diff, x_diff, d, x_n, y_n])  
            ts_start = ts_end
        

        if self._bc:
            # the end sequence will be (y_11,.., y_1n_C, y21, ..y_2n_C2, y1*, y1**, ...,y1****)    
            context_list = self.concat_context_multi_ts(inputs_for_processing) 
            print('context_list 0:', context_list[0])
            print('context_list 1:', context_list[1])


            x_emb, y, y_diff, x_diff, d, x_n, y_n  = context_list
        else:
            # the end sequence will be (y_11,.., y_1n_C, y21, ..y_2n_C, y1*, y2*, ..yk*, y1**, y2**, ..yk**)
            context_list = [self.concat_context_multi_ts(inputs_for_processing, j, n_C) for j in range(7)]
            target_list = [self.concat_target_multi_ts(inputs_for_processing, j, n_C, n_T, context_list[j].shape[-1]) for j in range(7)]
            x_emb, y, y_diff, x_diff, d, x_n, y_n = [tf.concat([context_list[j], target_list[j]], axis=1) for j in range(7)]
        inputs_for_processing = [x_emb, y, y_diff, x_diff, d, x_n, y_n]
        return inputs_for_processing, y, y_n, n_C, n_T, idx_c_all, idx_t_all





    def call(self, inputs):

        x, y, n_C, n_T, training, n_C_s, n_T_s, n_C_tot, n_T_tot = inputs #  (batch_size, n_C + n_T, 1), (batch_size, n_C + n_T, 1)

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
            inputs_for_processing, y, y_n, _, _, idx_c, idx_t = self.inputs_for_multi_ts(x, y, n_C, n_T, n_C_s, n_T_s)
            # print("y shape after inputs_for_processing:", y.shape)
            if (self._subsample):
                n_C = n_C_s
                n_T = n_T_s
                # print("n_C, n_T after subsample:", n_C, n_T)

        if self._bc: # currently w/o subsample
            n_C = n_C_tot
            n_T = n_T_tot
        else:
            n_C = n_C_tot
            n_T = n_T_tot
        inputs_for_processing.append(n_C)
        inputs_for_processing.append(n_T)
        print("n_C, n_T:", n_C, n_T)
        query_x, key_x, value_x, query_xy, key_xy, value_xy = self._feature_wrapper(inputs_for_processing) #  (batch_size, multiply *(n_C_s + n_T_s), enc_dim + multiply + x_diff.dim + x_n.dim), (batch_size, multiply *(n_C_s + n_T_s), enc_dim + multiply + x_diff.dim + x_n.dim) , (batch_size, multiply *(n_C_s + n_T_s), 1), (batch_size, multiply *(n_C_s + n_T_s), enc.dim + multiply + 2 + label.dim + y.dim + y_diff.dim + d.dim + y_n.dim)
        print("query_x shape:", query_x.shape)
        value_x = tf.reshape(value_x, (value_x.shape[0], (n_C + n_T), value_x.shape[-1]))
        y_n_closest = y_n[:, :, :y.shape[-1]] #### need to update this based on how we pick closest point
        ######## make mask #######
        print("y_n_closest shape:", y_n_closest.shape)
        mask = self._feature_wrapper.masker(n_C, n_T)
        print("mask shape:", mask.shape)
        μ, log_σ = self._atp([query_x, key_x, value_x, query_xy, key_xy, value_xy, mask, y_n_closest], training=training)

        return μ[:, n_C:], log_σ[:, n_C:], y[:, n_C:], idx_c, idx_t
      

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


