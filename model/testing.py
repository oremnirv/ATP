import tensorflow as tf
from Tutorials.helper import batcher
from model import losses
from model import atp_graph   
from model import atp_pipeline
import numpy as np
import time
import os



# save_dir = "/users/omernivron/Documents/ATP/weights/forecasting/weather"

def testing_multiple_times(inputs, save_dir, times=5, **kwargs):
    """
    **kwargs are the parameters for the model pipline in a dictionary format.
    """
    nll_list = []; mse_list = []
    num_epochs = 2;num_batches = 10; test_batch_s = 100
    num_heads=4; projection_shape_for_head=4; output_shape=64; rate=0.1; permutation_repeats=1;
    bound_std=False; num_layers=3; enc_dim=32; xmin=0.1; xmax=2
    for key, value in kwargs.items():
        if key == 'rate': rate = value
        elif key == 'output_shape': output_shape = value
        elif key == 'projection_shape_for_head': projection_shape_for_head = value
        elif key == 'num_heads': num_heads  = value
        elif key == 'permutation_repeats': permutation_repeats = value
        elif key == 'bound_std': bound_std = value
        elif key == 'num_layers': num_layers = value
        elif key == 'enc_dim': enc_dim = value
        elif key == 'xmin': xmin = value
        elif key == 'xmax': xmax = value
    for run_num in range(50, 50 + times):
        training_data_scaled, test_data_scaled = inputs
        step = 1
        run=run_num; 
        tf.random.set_seed(run)
        opt = tf.keras.optimizers.Adam(3e-4)
        atp_model = atp_pipeline.atp_pipeline( num_heads=num_heads, projection_shape_for_head=projection_shape_for_head, output_shape=output_shape, rate=rate, permutation_repeats=permutation_repeats,
        bound_std=bound_std, num_layers=num_layers, enc_dim=enc_dim,  xmin=xmin, xmax=xmax)
        tr_step = atp_graph.build_graph()
        name_comp = 'run_' + str(run)
        folder = save_dir + '/ckpt/check_' + name_comp
        if not os.path.exists(folder): os.mkdir(folder)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=atp_model)
        manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=1)
        ckpt.restore(manager.latest_checkpoint) 

        sum_mse_tot = 0; sum_nll_tot = 0
        for epoch in range(num_epochs):
            print("epoch: ",+epoch)
            for _ in range(num_batches):
                x,y = batcher(training_data_scaled[:,:1],training_data_scaled[:,-2:])
                n_C = int(np.random.choice(np.arange(2, 20), 1))
                n_T = 288 - n_C
                _,_, _, _ = tr_step(atp_model, opt, x,y,n_C,n_T, training=True)
        manager.save()
        step += 1
        ckpt.step.assign_add(1)
        for _ in range(test_data_scaled.shape[0] // test_batch_s):
            n_C = 10
            n_T = 200 - n_C
            t_te,y_te = batcher(test_data_scaled[:,:1], test_data_scaled[:,-2:], batch_s = test_batch_s)
            μ, log_σ = atp_model([t_te,y_te,10,200, False])
            _, sum_mse, sum_nll, _, _ = losses.nll(y_te[:, n_C:n_C+n_T], μ[:, n_C:], log_σ[:, n_C:])
            sum_nll_tot += sum_nll # each sequence is length 190 times 100 seq. per batch 
            sum_mse_tot += sum_mse
            
        nllx =  sum_nll_tot / (n_T * test_batch_s * (test_data_scaled.shape[0] // test_batch_s))
        msex =  sum_mse_tot / (n_T * test_batch_s * (test_data_scaled.shape[0] // test_batch_s))
        
            
        nll_list.append(nllx.numpy())
        mse_list.append(msex.numpy())
    np.save(save_dir + '/nll_list.npy', nll_list)    
    np.save(save_dir + '/mse_list.npy', mse_list)   