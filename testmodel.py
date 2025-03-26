import tensorflow as tf
from keras.models import load_model # type: ignore
import numpy as np
from PIL import Image, ImageOps
import os, pathlib, h5py
def WOW():
    np.set_printoptions(suppress=True)
    # Load the TFLite model
    model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), r"converted_tflite_quantized\keras_Model.h5")# Load the model path

    model = load_model(model_path, compile=False)# Load the model
    # Load labels
    labels_path = os.path.join(pathlib.Path(__file__).parent.absolute(), r"converted_tflite_quantized\labels.txt")
    with open(labels_path, "r") as f: #open label paths
        labels = [line.strip() for line in f.readlines()] #read the labels

    # Preprocess the image
    def preprocess_image(image_path):
        image = Image.open(image_path).convert("RGB")# Load the image and convert it to RGB
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)  # resizing the image to be at least 224x224 and then cropping from the center
        image_array = np.asarray(image) # turn the image into a numpy array
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1 # Normalise the array
        return normalized_image_array

    # Perform inference
    def predict(image_path):
        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32) # Create an array of the right shape to feed into the keras model
        image = preprocess_image(image_path) 
        data[0] = image # Load the image into the array
        
        
        prediction = model.predict(data) # Perform the prediction
        index = np.argmax(prediction) # Get the index of the highest probability
        predicted_label = labels[index] # Get the label with the highest probability
        confidence = prediction[0][index] # Get the confidence of the prediction

        return predicted_label, confidence # Return the predicted label and confidence

    # Test the model
    image_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "testimg1") # Load the image path
    predicted_label, confidence = predict(image_path) # Perform the prediction

    return predicted_label
    #print(f"Predicted label: {predicted_label}") # Print the predicted label
    #print(f"Confidence: {confidence:.2f}") # Print the confidence of the prediction

WOW() # Run the function