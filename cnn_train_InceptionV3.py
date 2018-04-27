import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
import argparse
import cv2
import os
import tensorflow as tf
import glob
import math
import pickle
import datetime
from load import data_process
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


def get_im(path, img_rows, img_cols):
    
    img = cv2.imread(path, 1)
    
    resized = cv2.resize(img, (img_rows, img_cols))  #size reduction
    return resized

def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')

def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def process2(f, img_rows, img_cols):
    X_train = []
    y_train = []
    print('reading..')
    with open(f) as file:
        file.readline()
        for row in file:
            row = row.strip().split(',')
            img = get_im(os.path.join('train-jpg', str(row[1])+'.jpg' ), img_rows, img_cols)
            
            X_train.append(img)
            y_train.append(row[3:])
    return X_train, y_train


def process3(n_classes, img_rows, img_cols, X_data, data_gt):
    X_data = np.array(X_data, dtype=np.uint8)
    data_gt = np.array(data_gt, dtype=np.uint8)
    X_data = X_data.astype('float32')
    X_data /= 255
    print("1..Shape of the data")
    print(X_data.shape)
    
    if K.image_data_format() == 'channels_first':
        X_data  = X_data.reshape(train_data.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        X_data = X_data.reshape(X_data.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
    #print('Train shape: ', X_data.shape)
    #print('train samples: ', X_data.shape[0])
    #print('train samples: ', data_gt.shape[0])
    return X_data, data_gt, input_shape

def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir('/Users/CASH/Deep_env/Amazon_MI/cacheV3'):
        os.mkdir('/Users/CASH/Deep_env/Amazon_MI/cacheV3')
    open(os.path.join('/Users/CASH/Deep_env/Amazon_MI/cacheV3', 'architecture.json'), 'w').write(json_string)
    model.save_weights(os.path.join('/Users/CASH/Deep_env/Amazon_MI/cacheV3', 'model_weights.h5'), overwrite=True)


def read_model():
    model = model_from_json(open(os.path.join('/Users/CASH/Deep_env/Amazon_MI/cacheV3', 'architecture.json')).read())
    model.load_weights(os.path.join('/Users/CASH/Deep_env/Amazon_MI/cacheV3', 'model_weights.h5'))
    return model


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(17, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model

def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
        model.compile(optimizer='rmsprop',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    return model

def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top
    layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in
    the inceptionv3 architecture
    Args:
    model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
        for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
            layer.trainable = True
            model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                              loss='categorical_crossentropy')
    return model

def main():
    
    #init image size and no of classes
    img_rows, img_cols = 32, 32
    n_classes = 17
    
    X_train = []
    y_train = []
    print('Read train images')
    cache_path = os.path.join('/Users/CASH/Deep_env/Amazon_MI/cacheV3', 'train.dat')
    if not os.path.isfile(cache_path):
        labels_l, df, labels_set = data_process()
        filename = 'one_hot.csv'
        X_data, data_gt = process2(filename, img_rows, img_cols)
        #convert img to np array
        X_data, data_gt, input_shape = process3(n_classes, img_rows, img_cols, X_data, data_gt)
        cache_data((X_data, data_gt, input_shape), cache_path)
    else:
        print('Restore train from cache!')
        X_data, data_gt, input_shape = restore_data(cache_path)
        print('Done restoring...')
    

    
    
    datagen = ImageDataGenerator(samplewise_std_normalization=True,
                                             rotation_range=90,
                                             horizontal_flip=True,
                                             vertical_flip=True)
    print('datagen...')
    datagen.fit(X_data)
    print('datagen.fit...')
    #create split for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_data, data_gt,  test_size=0.15)
    print('split...')
    nb_epoch = 10
    batch_size = 50
    print(X_test.shape)
    if not os.path.isfile('/Users/CASH/Deep_env/Amazon_MI/cacheV3/architecture.json'):
        print('model layers')
        base_model= InceptionV3(weights='imagenet', include_top=False)
        model = add_new_last_layer(base_model, n_classes)
        
        #model = setup_to_finetune(model)
        save_model(model)
    else:
        print('loading model layers')
        model=read_model()
    print('compliling model layers')

    print('fiting layers')
    base_model= InceptionV3(weights='imagenet', include_top=False)
    model = setup_to_transfer_learn(model, base_model)
    info = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=len(X_train),
                                  verbose=1,
                                  epochs=nb_epoch,
                                  class_weight='auto', validation_data=(X_test, y_test))
    print('fit..')
    score = model.evaluate(X_test, y_test, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print('\nploting..')
    print(info.history.keys())
    # summarize history for accuracy
    plt.plot(info.history['acc'])
    plt.plot(info.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()
    # summarize history for loss
    plt.plot(info.history['loss'])
    plt.plot(info.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



if __name__  == "__main__":
	main()
