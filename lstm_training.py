from util import dataset, plot_training, save_results
import numpy as np
from models import LSTM
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
resource = 'cpu'
cluster = 'f'
winsize = 144
h = 2

experiment_name = 'talos-LSTM_CNN-'+resource+'-'+cluster+'-w'+str(winsize)+'-h' + str(h)

# Data creation and load
ds = dataset.Dataset(meta=False, filename='res_task_'+cluster+'.csv', winSize=winsize, horizon=h, resource=resource)
print(ds.name)
ds.dataset_creation()
ds.data_summary()

model = LSTM.LSTMPredictor()
model.name = experiment_name

p = {'first_conv_dim': 32,
     'first_conv_activation': 'relu',
     'first_lstm_dim': 16,
     'second_lstm_dim': 16,
     'first_dense_dim': 8,
     'first_dense_activation': 'relu',
     'second_dense_dim': 8,
     'second_dense_activation': 'relu',
     'third_dense_dim': 1,
     'conv_kernel_init': 'he_normal',
     'dense_layers': 1,
     'batch_size': 128,
     'epochs': 1000,
     'patience': 50,
     'optimizer': 'adam',
     'batch_normalization': True,
     'lr': 1E-2,
     'momentum': 0.99,
     'decay': 1E-3,
     'pred_steps': 100
     }

train_model, history, forecast = model.training_talos(ds.X_train, ds.y_train, ds.X_test, ds.y_test, p)

plot_training.plot_series(np.arange(0, len(ds.y_test) - 1), ds.y_test, forecast, label1="ground truth",
                          label2="prediction", title=model.name)

plot_training.plot_loss(history, model.name)

plot_model(train_model, to_file='img/models/model_plot_' + model.name + '.png', show_shapes=True, show_layer_names=True)

save_results.save_output_csv(forecast, np.concatenate(ds.y_test[:len(forecast)], axis=0), 'avg' + resource, model.name)

save_results.save_params_csv(p, model.name)
