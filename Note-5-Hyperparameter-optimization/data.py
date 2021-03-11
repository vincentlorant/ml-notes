import cv2
import numpy as np
import os

def random_shape(config, img, r_shapes, i):
    '''Draw rectanles or circles
    '''
    if(r_shapes[i] == 0):
        coord = np.random.randint(config.IMG_SIZE, size=(1, 2))
        size = np.random.randint(10, size=1)

        cv2.circle(img,(
            coord[0][0],coord[0][1]),
                   size[0] +10,
                   (255,255,255), -1)

    elif(r_shapes[i] == 1):
        center = np.random.randint(config.IMG_SIZE//2, size=2)+30
        coord = np.random.randint(config.IMG_SIZE//3, size=4)

        cv2.rectangle(img,
                      (center[0] - coord[0],center[1] - coord[1]),
                      (center[0] + coord[2],center[1] + coord[3]),
                      (255,255,255),
                      thickness  = 1)


def get_data(config, nbr = 60):
    '''Build toy dataset
    '''
    r_shapes = np.random.randint(2, size=nbr)
    y_shape = list(r_shapes)# KU.to_categorical(list(r_shapes))
    data = np.empty((0,config.IMG_SIZE,config.IMG_SIZE,3))

    for i in range(nbr):
        img = np.zeros((config.IMG_SIZE,config.IMG_SIZE,3), np.uint8)
        random_shape(config, img, r_shapes, i)
        img = np.expand_dims(img, 0)
        data = np.append(data, img, axis=0)

    data= data/255
    return data, r_shapes

def create_dataset(config):
    dic = {'train':100, 'test':50}
    for set in ['train', 'test']:
        x, y = get_data(config, dic[set])
        np.save('data/x_'+set, x)
        np.save('data/y_'+set, y)



def data():
    path = 'D:/DATA/Works/AI/100_Posts/hyperparameter-tunning/data'
    x_train = np.load(os.path.join(path,'x_train.npy'))
    y_train = np.load(os.path.join(path,'y_train.npy'))
    x_test = np.load(os.path.join(path,'x_test.npy'))
    y_test = np.load(os.path.join(path,'y_test.npy'))

    return x_train, y_train, x_test, y_test
