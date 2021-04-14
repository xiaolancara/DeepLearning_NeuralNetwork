
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time


DATADIR = r'C:\AI program\2ndSemester\Numerical Methods\Deep Learning Neural Network\4th week_cats_dogs\datasets\PetImages'
CATEGORIES = ['Dog','Cat']
IMG_SIZE = 100

training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE,1)
y = np.array(y)

pickle_out = open('X.pickle','wb')
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open('y.pickle','wb')
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in_X = open('X.pickle','rb')
pickle_in_y = open('y.pickle','rb')
X = pickle.load(pickle_in_X)
y = pickle.load(pickle_in_y)

X = X/255.0 # normalize

dense_layers = [0]
layer_sizes = [32]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time()))

            print(NAME)
            model = Sequential()

            # one lyer
            model.add(Conv2D(layer_size,(3,3),input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size = (2,2)))

            for l in range(conv_layer-1):
                # two lyer
                model.add(Conv2D(layer_size,(3,3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size = (2,2)))

            model.add(Flatten())  # this converts our 3D feature map to 1D feature vectors
            for l in range(dense_layer):
                # three lyer
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            # output lyer
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

            model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])

            model.fit(X, y,
                      batch_size=32,
                      validation_split=0.3,
                      callbacks=[tensorboard]) # , epochs = 3
model.save('64x3-CNN.model')