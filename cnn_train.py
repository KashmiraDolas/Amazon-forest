import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from keras.optimizers import SGD
from mlxtend.evaluate import confusion_matrix

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
from sklearn.metrics import log_loss, label_ranking_average_precision_score


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
    if not os.path.isdir('/Users/CASH/Deep_env/Amazon_MI/cache32'):
        os.mkdir('/Users/CASH/Deep_env/Amazon_MI/cache32')
    open(os.path.join('/Users/CASH/Deep_env/Amazon_MI/cache32', 'architecture.json'), 'w').write(json_string)
    model.save_weights(os.path.join('/Users/CASH/Deep_env/Amazon_MI/cache32', 'model_weights.h5'), overwrite=True)


def read_model():
    model = model_from_json(open(os.path.join('/Users/CASH/Deep_env/Amazon_MI/cache32', 'architecture.json')).read())
    model.load_weights(os.path.join('/Users/CASH/Deep_env/Amazon_MI/cache32', 'model_weights.h5'))
    return model

def main():
    
    #init image size and no of classes
    img_rows, img_cols = 128, 128
    n_classes = 17
    
    X_train = []
    y_train = []
    print('Read train images')
    cache_path = os.path.join('/Users/CASH/Deep_env/Amazon_MI/cache32', 'train.dat')
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
    nb_epoch = 100
    batch_size = 50
    print(X_test.shape)
    if not os.path.isfile('/Users/CASH/Deep_env/Amazon_MI/cache32/architecture.json'):
        print('model layers')
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        print(model.output_shape)
        #model.add(Conv2D(64, (1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        print(model.output_shape)
        model.add(Conv2D(16, (1, 1), activation='relu'))
        print(model.output_shape)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        print(model.output_shape)
        model.add(Conv2D(32, (1, 1), activation='relu'))
        print(model.output_shape)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        print(model.output_shape)
        model.add(Conv2D(64, (1, 1), activation='relu'))
        print(model.output_shape)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        print(model.output_shape)
        model.add(Conv2D(1, (1, 1), activation='relu'))
        print(model.output_shape)
        model.add(Dropout(0.25))
        print(model.output_shape)
        model.add(Flatten())
        print(model.output_shape)
        model.add(Dense(128, activation='relu'))
        print(model.output_shape)
        model.add(Dense(n_classes, activation='softmax'))
        print(model.output_shape)
        save_model(model)
    else:
        print('loading model layers')
        model=read_model()
    print('\ncompliling model layers')

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    print('\n\nfiting layers')


    info =model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=(X_test, y_test))
    print(info)
    print(info.history)
    print('\n\nfit..')
    #model.fit(X_train, y_train,
    #  batch_size=batch_size,
    #      epochs=nb_epoch,
    #      verbose=1,
    #      validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=1)

    print('\nfit y_pred2')
    y_pred2 = model.predict(X_test,  verbose= 1)
    print('\n\nlabel_ranking_average_precision_score pred2')
    print(label_ranking_average_precision_score(y_test, y_pred2))
    
    print('\n\nTest loss:', score[0])
    print('\nTest accuracy:', score[1])
    
    print('\nploting..')
    print(info.history['acc'])
    # summarize history for accuracy
    acc = info.history['acc']
    acc.insert(0, 0)
    vacc = info.history['val_acc']
    vacc.insert(0, 0)
    plt.plot(acc)
    plt.plot(vacc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.margins(x=0, y = 0)

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


    info =model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),steps_per_epoch=len(X_train) / batch_size, verbose=1,epochs=nb_epoch,validation_data=(X_test, y_test))
    print(info)
    print(info.history)
    print('\n\nfit..')
    #model.fit(X_train, y_train,
    #  batch_size=batch_size,
    #      epochs=nb_epoch,
    #      verbose=1,
    #      validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=1)

    print('\nfit y_pred2')
    y_pred2 = model.predict(X_test,  verbose= 1)
    print('\n\nlabel_ranking_average_precision_score pred2')
    print(label_ranking_average_precision_score(y_test, y_pred2))

    print('\n\nTest loss:', score[0])
    print('\nTest accuracy:', score[1])

    print('\nploting..')
    print(info.history.keys())
    # summarize history for accuracy
    plt.plot(info.history['acc'])
    plt.plot(info.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.margins(x=0, y = 0)
    plt.ylim((0, 100))
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
