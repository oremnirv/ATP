from model import atp_graph, losses
from data import synthetic_data_gen, feature_extractor
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from model import atp_pipeline
from data import dataset_preparer
import argparse
from Tutorials.helper import batcher
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str, help="dataset")
    # parser.add_argument("type_model", type=str, help="type of np model")
    parser.add_argument("iterations", type=int, help="number of iterations for training")
    parser.add_argument("num_repeats", type=int, help="number of random seed repeats")
    parser.add_argument("n_C",type=int,help = "context")
    parser.add_argument("n_T",type=int,help = "target")


    args = parser.parse_args()

    if args.dataset == "weather":
        weather_processor = dataset_preparer.weather(path_to_weather_data="datasets/weather.csv")
        x_train, y_train, x_val, y_val, x_test, y_test = weather_processor(0.1) 
        save_dir = "weights/forecasting/weather"
    
    n_C = args.n_C
    n_T = args.n_T

    opt = tf.keras.optimizers.Adam(3e-4)
    batch_size = 32

    for i in range(args.num_repeats):

        step = 1
        run= 50 + i
        tf.random.set_seed(run)
        atp_model = atp_pipeline.instantiate_model(args.dataset)
        tr_step = atp_graph.build_graph()
        name_comp = 'run_' + str(run)
        folder = save_dir + '/ckpt/check_' + name_comp
        if not os.path.exists(folder): os.mkdir(folder)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=atp_model)
        manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=1)
        ckpt.restore(manager.latest_checkpoint) 
        sum_mse_tot = 0; sum_nll_tot = 0
        mini = 50000
        
        for i in range(args.iterations):
            idx_list = len(range(x.shape[0]))
            x,y,_ = batcher(x_train,y_train,idx_list,window=n_C+n_T) ####### generalise for not weather
            _,_, _, _ = tr_step(atp_model, opt, x,y,n_C,n_T, training=True)

            if i % 100 == 0:
                idx_list = len(range(x_val.shape[0]))
                t_te,y_te,_ = batcher(x_val,y_val,idx_list,batch_s = 100,window=n_C+n_T)
                μ, log_σ = atp_model([t_te, y_te, n_C, n_T, False])
                _,_,_, nll_pp_te, msex_te = losses.nll(y_te[:, n_C:n_C+n_T], μ, log_σ)

                if nll_pp_te < mini:
                    mini = nll_pp_te
                    manager.save()
                    step += 1
                    ckpt.step.assign_add(1)

        ckpt.restore(manager.latest_checkpoint) 
        #### test the restore function ####
   
        idx_list = list(range(x_test.shape[0]))

        for _ in range(x_test.shape[0]//test_batch_s): #### specify correct number of batches for the batcher #####
            
            t_te,y_te,idx_list = batcher(x_test,y_test, idx_list,batch_s = test_batch_s,window=n_C+n_T)
            μ, log_σ = atp_model([t_te,y_te,n_C,n_T, False])
            _, sum_mse, sum_nll, _, _ = losses.nll(y_te[:, n_C:n_C+n_T], μ, log_σ)
            sum_nll_tot += sum_nll / n_T # each sequence is length 190 times 100 seq. per batch 
            sum_mse_tot += sum_mse / n_T
            
        nllx =  sum_nll_tot / (test_batch_s * x_test.shape[0]//test_batch_s)
        msex =  sum_mse_tot / (test_batch_s * x_test.shape[0]//test_batch_s)
        
            
        nll_list.append(nllx.numpy())
        mse_list.append(msex.numpy())
            
        np.save(save_dir + '/nll_list.npy', nll_list)    
        np.save(save_dir + '/mse_list.npy', mse_list)  


