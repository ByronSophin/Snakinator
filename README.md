# Snake Species Classifier

## Explanation & Overview

The **Snake Species Classifier** project uses Convolutional Neural Networks (CNNs) to predict the species of a snake from an image input. The model was trained on a dataset obtained from Kaggle. The training process involved experimenting with several CNN models to find the best architecture for accurate predictions. The trained model is deployed using **Streamlit**, which serves as a simple frontend, allowing users to upload snake images and receive predictions on the species.

## Features Implemented

- Image Upload: Users can upload a snake image.
- Prediction: The AI predicts the species of the snake in the image.
- Multiple Models: The model supports various versions, such as `V2.0`, `V4.1`, and `V5`.
- Data Augmentation: The model incorporates techniques like rotation, width/height shift, and horizontal flip to enhance model performance.
- Model Saving and Loading: The trained models are saved and loaded for inference with `keras`.
- Result Display: The predicted species name is displayed to the user based on the uploaded image.

## Technologies Used

- Python: Core programming language for model building and deployment.
- Keras: Framework for building Convolutional Neural Networks.
- TensorFlow: Backend library for model training and inference.
- Streamlit: Framework for creating the web app frontend.
- OpenCV, Matplotlib: Used for image preprocessing and visualization.

## How to Get Started
- Clone the repository:
- git clone https://github.com/your-username/snake-species-classifier.git
- create virtual environment and install dependencies
- pip install tensorflow keras streamlit scikit-learn numpy matplotlib pandas opencv-python imageio h5py pillow
- To run the app, use the command streamlit run snake2.py

## Getting the model
- The model is created in the model.py files.
- Model relies on training and test data. The exact data set used is found via kaggle at https://www.kaggle.com/datasets/goelyash/165-different-snakes-species/data. 
