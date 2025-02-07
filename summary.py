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
model1 = load_model('snake_modelV3.1.keras')
model1.summary()