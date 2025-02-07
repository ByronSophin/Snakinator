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
from keras.applications import EfficientNetV2L
from sklearn.model_selection import train_test_split
import random
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

base_model = EfficientNetV2L(include_top=False, input_shape=(64, 64, 3), weights='imagenet')

# Freeze the layers of the base model
base_model.trainable = False

# Create a Sequential model
model = keras.models.Sequential()

# Add the base model to the sequential model
model.add(base_model)

# Flatten the output of the base model
model.add(Flatten())

# Add your own dense output layer
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(135, activation='softmax'))  # Assuming you have 135 classes

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Display model summary
model.summary()

history = model.fit(
    training_generator,
    epochs=10,
    validation_data=validation_generator,
)

model.save('snake_modelV5.0.keras')


