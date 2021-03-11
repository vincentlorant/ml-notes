import numpy as np
import os

import keras.utils as KU
import keras.layers as KL
import keras.utils as KU
from keras.models import Model, Sequential
from keras.optimizers import *
from keras import metrics

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe


def data():
    path = 'D:/DATA/Works/AI/100_Posts/hyperparameter-tunning/data'
    x_train = np.load(os.path.join(path,'x_train.npy'))
    y_train = np.load(os.path.join(path,'y_train.npy'))
    x_test = np.load(os.path.join(path,'x_test.npy'))
    y_test = np.load(os.path.join(path,'y_test.npy'))

    return x_train, y_train, x_test, y_test



def create_model(x_train, y_train, x_test, y_test):
    Inputs = (64, 64, 3)
    model = Sequential()

    model.add(KL.Conv2D(32, kernel_size=(3, 3),input_shape=Inputs))
    model.add(KL.Activation('relu'))
    model.add(KL.BatchNormalization())
    model.add(KL.MaxPooling2D((2, 2)))
    model.add(KL.Dropout({{uniform(0, 1)}}))

    model.add(KL.Flatten())
    model.add(KL.Dense({{choice([16 ,32, 64, 128])}}))
    model.add(KL.Activation({{choice(['relu', 'sigmoid'])}}))


    if {{choice(['three', 'four'])}} == 'four':
        model.add(KL.Dense({{choice([16 ,32, 64, 128])}}))
        model.add(KL.Activation('relu'))


    model.add(KL.Dense(1))
    model.add(KL.Activation('sigmoid'))


    adam=Adam(lr={{uniform(0.00005,0.01)}})
    model.compile(optimizer=adam,
                      loss='binary_crossentropy',
                      metrics=[metrics.binary_accuracy])


    H = model.fit(
        x_train,
        y_train,
        epochs=6,
        batch_size={{choice([64, 128])}})

    score = model.evaluate(x_test, y_test, verbose=0)
    accuracy = score[1]
    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}



if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=25,
                                              trials=Trials())

    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
