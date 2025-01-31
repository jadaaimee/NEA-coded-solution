import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import os, pathlib
np.set_printoptions(suppress=True)
# Load the TFLite model
model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), r"converted_tflite_quantized\keras_Model.h5")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Load labels
labels_path = os.path.join(pathlib.Path(__file__).parent.absolute(), r"converted_tflite_quantized\labels.txt")
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)  # resizing the image to be at least 224x224 and then cropping from the center
    image_array = np.asarray(image) # turn the image into a numpy array
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1 # Normalise the array
    return normalized_image_array

# Perform inference
def predict(image_path):
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = preprocess_image(image_path=os.path.join(pathlib.Path(__file__).parent.absolute(), "test_image.jpg.jpg"))
    data[0] = image
    
    model = load_model(model_path, compile=False)
    prediction = model.predict(data)
    index = np.argmax(prediction)
    predicted_label = labels[index]
    confidence = prediction[0][index]

    return predicted_label, confidence

# Test the model
image_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "test_image.jpg.jpg")
predicted_label, confidence = predict(image_path)

print(f"Predicted label: {predicted_label}")
print(f"Confidence: {confidence:.2f}")

