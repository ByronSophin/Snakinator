import streamlit as sl
from keras.models import load_model
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
from keras.preprocessing import image
import numpy as np
import random
import csv

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

# Create the training generator to get the classes
training_generator = train_gen.flow_from_directory(
    train_dir,
    target_size=(224,224),  #does not need to be changed, no training is done with this file. only used for generating the class array/model prediction
    batch_size=batchSize,
    class_mode='categorical',
    subset='training'
)
classArray = training_generator.class_indices

def predict_snake_species(img_file): #function for loading the model and generating the prediction from a passed in input image
    Snakinator = load_model('snake_modelV3.5.keras')
    inputSize = Snakinator.input_shape[1:]
    
    csv_file = './csv/test.csv'

    # Create a dictionary used for mapping the species name to the number predicted by the actual model, mapping gotten from the csv file data
    numberSpecies = {}

    #Add csv contents to the dictionary for species and number
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first row the the attributes
        for row in reader:
            species_name = row[1] #this column is for species nomenclature
            number = row[7]  #this column is for the number associated with species name, same as subfolder names
            #check that the species name and number is not already in the dictionary
            if number not in numberSpecies:
                numberSpecies[number] = species_name

    #preprocessing
    img = image.load_img(img_file, target_size=inputSize)
    imgArray = image.img_to_array(img)
    imgArray = np.expand_dims(imgArray, axis=0)
    imgArray = imgArray / 255.0
    #generating the array of 135 with predicted values in each index
    predictions = Snakinator.predict(imgArray)

    #Get the predicted label index
    predictedIndex = np.argmax(predictions) #get the largest valued index from the array of predictions of the model. Highest value = prediction

    #Find the corresponding species name from the predicted index
    prediction = list(classArray.keys())[list(classArray.values()).index(predictedIndex)]

    return prediction, numberSpecies

def main(): #main function that generates the upload file, button, and prediction
    sl.title("Snake Classifier")
    img_file = sl.file_uploader("Choose an image of a Snake", type=["png", "jpg"])
    if img_file is not None:
        if sl.button("Predict"):
            sl.image(img_file, caption="Uploaded Image", use_column_width=True)
            prediction, numberSpecies = predict_snake_species(img_file)
            sl.write("Predicted species:", numberSpecies[prediction])

if __name__ == "__main__": #there were some conflicting functions
    main()