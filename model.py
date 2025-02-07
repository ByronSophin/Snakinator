import sys
import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
#import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import MobileNetV2, EfficientNetV2L, EfficientNetB7
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
#from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import load_model
import random
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
'''
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
'''
#seed 42 only
np.random.seed(42)
tf.random.set_seed(42)

labels = os.listdir('./train')
species = len(labels)
print(species)

trainDir = "./train/"
testDir = "./test/"
batchSize = 64 #change batchSize

#generator for training with data augmentation
train_gen = ImageDataGenerator(rescale = 1./255, 
                                validation_split=0.2,
                                rotation_range=20,
                                width_shift_range=0.2, 
                                height_shift_range=0.2,  
                                horizontal_flip=True) 

training_generator = train_gen.flow_from_directory(
    trainDir,
    target_size=(224,224), #change image size
    batch_size=batchSize,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_gen.flow_from_directory(
    trainDir,
    target_size=(224,224), #change image size
    batch_size=batchSize,
    class_mode='categorical',
    subset='validation'
)


#Model 4
'''
model = keras.models.Sequential()

model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(135, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
'''

#Model 5
'''
baseModel = EfficientNetV2L(include_top=False, input_shape=(64, 64, 3), weights='imagenet')

baseModel.trainable = False

model = keras.models.Sequential()

model.add(baseModel)

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(135, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
'''

#Model 6
'''
featureModel = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

featureModel.trainable = True

model = keras.models.Sequential()

model.add(featureModel)

model.add(GlobalAveragePooling2D())

model.add(keras.layers.Dense(200, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(keras.layers.Dense(150, activation="relu"))
model.add(BatchNormalization())
model.add(keras.layers.Dense(135, activation="softmax"))

optimizer = Adam(learning_rate=0.0001)
model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])


# Display the model archetecture
model.summary()
'''

model = load_model('snake_modelV3.4.keras')

history = model.fit(
    training_generator,
    epochs=5,
    validation_data=validation_generator,
)

model.save('snake_modelV3.5.keras')


