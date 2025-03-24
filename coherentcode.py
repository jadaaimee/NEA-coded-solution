import tensorflow as tf
from keras.models import load_model # type: ignore
import numpy as np
from PIL import Image, ImageOps
import os, pathlib, time
#import RPi.GPIO as GPIO # type: ignore

# ========== GLOBAL CONFIGURATIONS ==========
MODEL_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), "converted_tflite_quantized/keras_Model.h5")
LABELS_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), "converted_tflite_quantized/labels.txt")
IMAGE_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), "test_image.jpg")
PUMP_PIN = 17  # GPIO pin connected to the water pump

# ========== MODEL LOADING ==========
model = load_model(MODEL_PATH, compile=False)

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ========== IMAGE PREPROCESSING ==========
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1  # Normalize between -1 and 1
    return normalized_image_array

# ========== PREDICTION ==========
def predict(image_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = preprocess_image(image_path)
    data[0] = image

    prediction = model.predict(data)
    index = np.argmax(prediction)
    predicted_label = labels[index]
    confidence = prediction[0][index]

    return predicted_label, confidence

# ========== TIMER FUNCTIONS ==========
def last_photo_taken_timer(photo_taken_time):
    current_time = time.time()
    if photo_taken_time is not None:
        timer = current_time - photo_taken_time
        print(f"Time since last photo: {timer:.2f} seconds")
    else:
        print("No photo has been taken yet.")

def last_watered_timer(watered_time):
    current_time = time.time()
    if watered_time is not None:
        timer = current_time - watered_time
        hours = int(timer // 3600)
        minutes = int((timer % 3600) // 60)
        print(f"Plant was last watered: {hours} hours and {minutes} minutes ago.")
    else:
        print("Plant has not yet been watered.")

# ========== WATERING SYSTEM ==========
GPIO.setmode(GPIO.BCM)
GPIO.setup(PUMP_PIN, GPIO.OUT)

def release_water(last_watered_time):
    current_time = time.time()
    print("Checking if watering is needed...")

    # Water if more than 3 hours have passed
    if last_watered_time is None or (current_time - last_watered_time) >= (3 * 3600):
        print("Watering plant now...")
        GPIO.output(PUMP_PIN, GPIO.HIGH)
        time.sleep(5)  # Watering duration (adjust as needed)
        GPIO.output(PUMP_PIN, GPIO.LOW)
        print("Watering complete.")

        last_watered_time = current_time  # Update last watered time
    else:
        print("No watering needed yet.")

    return last_watered_time

# ========== MAIN EXECUTION ==========
def main():
    last_watered_time = None
    photo_taken_time = time.time()

    # Run prediction
    predicted_label, confidence = predict(IMAGE_PATH)
    print(f"Predicted label: {predicted_label} with confidence {confidence:.2f}")

    # Decision making
    if predicted_label == "Wilted" and confidence > 0.8:
        print("The plant is wilted and needs watering.")
        last_watered_time = release_water(last_watered_time)
    elif predicted_label == "Healthy":
        print("The plant is healthy and does not need watering.")

    # Timers
    last_photo_taken_timer(photo_taken_time)
    last_watered_timer(last_watered_time)

    # Cleanup GPIO
    GPIO.cleanup()

if __name__ == "__main__":
    main()
