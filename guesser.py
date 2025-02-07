import sys
import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
#import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import random
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

Snakinator = load_model('snake_modelV4.1.keras')

import csv

# Define the path to the CSV file
csv_file = './csv/test.csv'

# Initialize an empty dictionary to store the mapping
species_mapping = {}

# Read the CSV file and populate the dictionary
with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row if it exists
    for row in reader:
        species_name = row[1]  # Assuming species name is in the first column
        number = row[7]  # Assuming associated number is in the second column
        # Check if the species name and number are different
        if number not in species_mapping:
            species_mapping[number] = species_name

imgPath = './images/Agkistrodon_contortrix1.png'
img = image.load_img(imgPath, target_size=(64,64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Make predictions
predictions = Snakinator.predict(img_array)

# Load the class indices from the training data
# Define the directories for training and testing data
train_dir = "./train/"
test_dir = "./test/"

# Define the batch size
batchSize = 64

# Create an ImageDataGenerator for preprocessing images
train_gen = ImageDataGenerator(rescale=1./255,
                               validation_split=0.2,
                               rotation_range=20,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               horizontal_flip=True)

# Create the training generator
training_generator = train_gen.flow_from_directory(
    train_dir,
    target_size=(64,64),  # Use the same target size as in the prediction file
    batch_size=batchSize,
    class_mode='categorical',
    subset='training'
)

class_indices = training_generator.class_indices

# Get the predicted label index
predicted_label_index = np.argmax(predictions)

# Map the predicted label index to the corresponding species name
predicted_species = list(class_indices.keys())[list(class_indices.values()).index(predicted_label_index)]


print("Predicted species:", species_mapping[predicted_species])