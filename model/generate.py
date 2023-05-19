from data_wrangler import dataset_preparer
from model import model_init
import numpy as np
import matplotlib.pyplot as plt
import IPython


def generate(model, y, n_C, n_T, μ=[], sample=True, num_samples=20, show=False):
    x = np.linspace(-1, 1, n_C + n_T)[np.newaxis, :, np.newaxis]
    y_temp = y[:,:n_C,:].copy().reshape(1, -1, 1) 
    y_temp = np.repeat(y_temp, num_samples, axis=0)
    x_temp = np.repeat(x, num_samples, axis=0)
    for gen_num in range(n_T):
        y_temp = np.concatenate([y_temp, np.zeros((num_samples, 1, 1))], axis=1)
        if show:
            IPython.display.clear_output(wait=True)            
            plt.plot(y[0, :n_C + n_T,:].reshape(-1), color="blue")
            for i in range(num_samples):
                plt.scatter(np.arange(y_temp.shape[1]), y_temp[i, :y_temp.shape[1], :].reshape(-1))
            plt.show() 
        
        μ_tp1, log_σ_tp1  = model([x_temp[:, :y_temp.shape[1], :], y_temp, n_C, 1 + gen_num, False]) 
        if sample:
            μ.append(np.random.normal(μ_tp1.numpy()[:, -1, 0], np.exp(log_σ_tp1.numpy()[:, -1, 0])))
        else:
            μ.append(μ_tp1.numpy()[:, -1, 0])
        
        y_temp[:, -1, :] =  μ[-1][:, np.newaxis]

    return y_temp, μ

if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, y_test = dataset_preparer.dataset_processor(path_to_data="datasets/ETTm2.csv") 
    m = model_init.model_init('atp', 0, 'forecasting/ETT/')
    m.load_model()
    seq, μ = generate(m.model, y_train[:1, :, :], n_C=96, n_T=25, μ=[], sample=True, num_samples=5, show=False)
