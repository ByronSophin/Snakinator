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
from keras.applications import EfficientNetB7
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import random
from keras.models import load_model
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

np.random.seed(42)
tf.random.set_seed(42)

labels = os.listdir('./train')
species = len(labels)
print(species)


train_dir = "./train/"
test_dir = "./test/"
batchSize=64

train_gen = ImageDataGenerator(rescale = 1./255, 
                                validation_split=0.2,
                                rotation_range=20,  # Data augmentation: random rotation
                                width_shift_range=0.2,  # Data augmentation: horizontal shift
                                height_shift_range=0.2,  # Data augmentation: vertical shift
                                horizontal_flip=True)  # Data augmentation: horizontal flip)

training_generator = train_gen.flow_from_directory(
    train_dir,
    target_size=(64,64),
    batch_size=batchSize,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_gen.flow_from_directory(
    train_dir,
    target_size=(64,64),
    batch_size=batchSize,
    class_mode='categorical',
    subset='validation'
)
'''
# Define the model
model = keras.models.Sequential()

# Convert input range from 0-255 to 0-1
model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Concatenate max-pooling layers
model.add(Flatten())

# Fully connected layer
model.add(Dense(256, activation='relu'))

# Dropout layer
model.add(Dropout(0.4))

# Output layer
model.add(Dense(135, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Display model summary
model.summary()
'''
model = load_model('snake_modelV4.1.keras')
history = model.fit(
    training_generator,
    epochs=50,
    validation_data=validation_generator,
)

model.save('snake_modelV4.2.keras')


