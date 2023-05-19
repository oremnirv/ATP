import os
import tensorflow as tf
from model import atp_graph
from model import atp_pipeline
from data_wrangler import dataset_preparer, batcher

class model_init():
        def __init__(self, model_name, run, task='forecasting/ETT/'):
            super().__init__()
            self.model_name = model_name
            self.opt = tf.keras.optimizers.Adam(3e-4)
            self.run = run
            self.save_dir = "weights/{}/{}/".format(task, model_name)
            self.tr_step = atp_graph.build_graph()
            assert model_name in ['atp_no_leakage_new_block', 'atp', 'new_block']
            if model_name == 'atp_no_leakage_new_block':
                pass
            elif model_name == 'atp':
                self.model = atp_pipeline.atp_pipeline(num_heads=6, projection_shape_for_head=11, output_shape=32, rate=0.05, permutation_repeats=0,
                    bound_std=False, num_layers=4, enc_dim=32, xmin=0.1, xmax=1,MHAX_leakage="xxx")  
            elif model_name == 'new_block':
                self.model = atp_pipeline.atp_pipeline(num_heads=10, projection_shape_for_head=9, output_shape=32, rate=0.0, permutation_repeats=0,
                    bound_std=False, num_layers=6, enc_dim=32, xmin=0.1, xmax=2,MHAX_leakage="new_block")
            else:
                raise ValueError('model_name not found')

        def load_model(self):
            name_comp = 'run_' + str(self.run)
            folder = self.save_dir + '/ckpt/check_' + name_comp
            if not os.path.exists(folder): os.mkdir(folder)
            ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.opt, net=self.model)
            manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=1)
            ckpt.restore(manager.latest_checkpoint) 
            return  ckpt, manager
        

if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, y_test = dataset_preparer.dataset_processor(path_to_data="datasets/ETTm2.csv") 
    m = model_init('atp', 0, 'forecasting/ETT/')
    m.load_model()
    n_C = 96
    n_T = 192
    idx_list = list(range(x_val.shape[0] - (n_C+n_T)))
    x,y,_, _ = batcher.batcher(x_val,y_val,idx_list,window=n_C+n_T)
    μ, log_σ = m.model([x, y, n_C, n_T, False])