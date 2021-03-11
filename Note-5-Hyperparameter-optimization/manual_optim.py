import cv2
import numpy as np
import os

import keras.utils as KU
import keras.layers as KL
import keras.utils as KU
from keras.models import Model, Sequential
from keras.optimizers import *
from keras import metrics



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
    model.add(KL.Dropout(0.5))

    model.add(KL.Flatten())
    model.add(KL.Dense(64))
    model.add(KL.Activation('relu'))

    model.add(KL.Dense(1))
    model.add(KL.Activation('sigmoid'))


    model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=[metrics.binary_accuracy])

    H = model.fit(
        x_train,
        y_train,
        epochs=6,
        batch_size=128)

    H = model.evaluate(x_test, y_test)
    print('loss and binary accuracy on test samples', H)



def train_model():
    x_train, y_train, x_test, y_test = data()
    create_model(x_train, y_train, x_test, y_test)



if __name__ == '__main__':
    main()
