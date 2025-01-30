import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TFLite model
model_path = r"C:\Users\User\OneDrive - Badminton School\Documents\A-LEVELS\A-LEVEL COMPUTER SCIENCE\UNIT 2\NEA PROJECT\NEA coded solution\converted_tflite_quantized (1)\model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Load labels
labels_path = r"C:\Users\User\OneDrive - Badminton School\Documents\A-LEVELS\A-LEVEL COMPUTER SCIENCE\UNIT 2\NEA PROJECT\NEA coded solution\converted_tflite_quantized (1)\labels.txt"
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).resize((224, 224))  # Resize to match the model's input size
    image = np.array(image).astype("float32") / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Perform inference
def predict(image_path):
    image = preprocess_image(image_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Ensure the input tensor type matches the model's expected type
    input_dtype = input_details[0]['dtype']
    print("Model expects dtype:", input_dtype)

    if input_dtype == np.uint8:
        image =image
    elif input_dtype == np.float32:
        image = image.astype(np.float32) / 255.0  

# Ensure batch dimension is added
    image = np.expand_dims(image, axis=0)

    print("After dtype conversion:", image.dtype, image.shape, image.min(), image.max())
    image = (image * 255).astype(np.uint8)
    input_tensor = input_tensor.reshape(224, 224, 3)
  # Adjust based on expected input shape


    # Feed the image into the model
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Get the prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = labels[np.argmax(output_data)]
    confidence = np.max(output_data)

    return predicted_label, confidence

# Test the model
image_path = r"C:\Users\User\OneDrive - Badminton School\Documents\A-LEVELS\A-LEVEL COMPUTER SCIENCE\UNIT 2\NEA PROJECT\NEA coded solution\IMG_1261.jpeg"
predicted_label, confidence = predict(image_path)

print(f"Predicted label: {predicted_label}")
print(f"Confidence: {confidence:.2f}")

