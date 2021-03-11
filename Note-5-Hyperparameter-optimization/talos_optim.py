import talos as ta
import wrangle as wr
from talos.metrics.keras_metrics import fmeasure_acc
from talos import live

import keras.utils as KU
import keras.layers as KL
import keras.utils as KU
from keras.models import Model, Sequential
from keras.optimizers import *
from keras.activations import relu, elu
from keras import metrics

import os
import numpy as np



def data():
    path = 'D:/DATA/Works/AI/100_Posts/hyperparameter-tunning/data'
    x_train = np.load(os.path.join(path,'x_train.npy'))
    y_train = np.load(os.path.join(path,'y_train.npy'))

    return x_train, y_train


def create_model(x_train, y_train, x_val, y_val, params):
    Inputs = (64, 64, 3)
    model = Sequential()

    model.add(KL.Conv2D(32, kernel_size=(3, 3),input_shape=Inputs))
    model.add(KL.Activation('relu'))
    model.add(KL.BatchNormalization())
    model.add(KL.MaxPooling2D((2, 2)))
    model.add(KL.Dropout(params['dropout']))

    model.add(KL.Flatten())
    model.add(KL.Dense(params['first_neuron']))
    model.add(KL.Activation(params['activation']))


    model.add(KL.Dense(1))
    model.add(KL.Activation('sigmoid'))


    model.compile(optimizer=params['optimizer'](),
                      loss='binary_crossentropy',
                      metrics=[metrics.binary_accuracy])


    H = model.fit(
    x_train,
    y_train,
    validation_data=[x_val, y_val],
    epochs=params['epochs'],
    batch_size=params['batch_size'])

    return H, model



def main():
    x, y = data()

    p = {'dropout': [0,0.25,0.5,0.75],
        'first_neuron':[16 ,32, 64, 128, 256],
        'activation':[relu, elu],
        'optimizer': [Nadam, Adam],
        'epochs': [6],
        'batch_size': [32, 64]}

    t = ta.Scan(x=x,
                y=y,
                model=create_model,
                params=p,
                dataset_name='shapes',
                grid_downsample=1,
                experiment_no='1')
    return t


if __name__ == '__main__':
    main()
