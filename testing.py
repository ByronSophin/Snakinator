import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

testingModel = load_model('snake_modelV3.5.keras')
#testingModel.input_shape
inputSize = testingModel.input_shape[1:3]
testDir = "./test/"

test_gen = ImageDataGenerator(rescale=1./255)

# Load test data using the image data generator
testing_generator = test_gen.flow_from_directory(
    testDir,
    target_size = inputSize,  
    batch_size= 32,
    class_mode='categorical',
    shuffle=False, 
    seed=42
)

# Evaluate the model on the test data
results = testingModel.evaluate(testing_generator)

print("Test Loss:", results[0])
print("Test Accuracy:", results[1])
