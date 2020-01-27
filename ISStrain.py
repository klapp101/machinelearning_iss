#alexalvertos

#ISStrain working model with instrument
# classification with convelutional neural networks


import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import cv2
from tqdm import tqdm
import random
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

DATADIR = "/Users/alexalvertos/Desktop/ISStrain/"

CATEGORIES = ["Flute","Harmonica","Bagpipes","Piano","Saxophone","Guitar",]
training_data = []
IMG_SIZE = 350
def train():
    for category in CATEGORIES:  # iterate each religion

        path = os.path.join(DATADIR,category)  # create path to different religions
        class_num = CATEGORIES.index(category)  # get the classification of each religion

        for img in tqdm(os.listdir(path)):  # iterate over each image in each religion
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:
                pass

train()

# Shuffling data so each religion is trained without bias
random.shuffle(training_data)
X = []
y = []

for religion_features,religion_label in training_data:
    X.append(religion_features)
    y.append(religion_label)

print("The size of X is ",len(X))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y=np.array(y)

#pickle_save_X = open("X.pickle","wb")
#pickle.dump(X, pickle_save_X)
#pickle_save_X.close()
#pickle_save_y = open("y.pickle","wb")
#pickle.dump(y, pickle_save_y)
#pickle_save_y.close()

#pickle_load_X = open("X.pickle","rb")
#X = pickle.load(pickle_load_X)
#pickle_load_y = open("y.pickle","rb")
#y = pickle.load(pickle_load_y)
#X = X/255.0

#model = Sequential()
#model.add(Conv2D(117, (3, 3), input_shape=X.shape[1:]))
#model.add(Activation('relu'))
#model.add(Conv2D(117, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
#model.add(Activation("relu"))
#model.add(Dense(6))
#model.compile(loss='sparse_categorical_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])

pickle_load_X = open("X.pickle","rb")
X = pickle.load(pickle_load_X)
pickle_load_y = open("y.pickle","rb")
y = pickle.load(pickle_load_y)
X = X/255.0
#datagen = ImageDataGenerator(
#    featurewise_center=True,
#    featurewise_std_normalization=True,
#    rotation_range=20,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    horizontal_flip=True)
#datagen.fit(X)
# fits the model on batches with real-time data augmentation:
model = Sequential()
model.add(Conv2D(20, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(20, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(6))
model.add(Activation("sigmoid"))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# Go through 10 samples at a time
# Train 70% , Test 30%
# Iterate 3 times (epochs = 3)
#model.fit_generator(datagen.flow(X, y, batch_size=32),
#                    steps_per_epoch=len(X)/32, epochs=3)
model.fit(X, y, batch_size=10, epochs=3, validation_split=0.3)
