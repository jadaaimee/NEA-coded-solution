import os, pathlib
import time
#make code to note time of photo taken
#make code to note time of plant watered

def LastPhotoTakenTimer(PhotoTakenTime):
    #get current time
    CurrentTime=time.time()

    if PhotoTakenTime is not None:
        timer=CurrentTime-PhotoTakenTime
        print(timer)
    else:
       print(None)

# #testing
# LastPhotoTakenTimer(time.time() - (3600 * 1))
# LastPhotoTakenTimer(time.time() - (1800 * 1))
# LastPhotoTakenTimer(time.time() - (3600 * 2 + 900))
# LastPhotoTakenTimer(time.time() - (60 * 5))
# LastPhotoTakenTimer(time.time() - (3600 * 3 + 2700))
# LastPhotoTakenTimer(None)

#last waterd timer fucntion
def LastWateredTimer(WateredTime):
    CurrentTime=time.time()

    if WateredTime is not None:
        timer = CurrentTime - WateredTime
        print(timer)
    else:
        print(None)

def LastWateredScreen(WateredTime):
    CurrentTime=time.time()
    if WateredTime is not None:
        timer = CurrentTime - WateredTime
        hours = int(timer//3600)
        minutes = int((timer%3600)//60)
    #i have changed this from showing the exact time, to showing how long ago, to remove confusion
        print(f"Plant was last watered: {hours} hours and {minutes} minutes ago.")
    else:
        print("Plant has not yet been watered")

# #testing
# LastWateredTimer(time.time()-(3600*1+900))
# LastWateredTimer(time.time()-(3600*3+2700))
# LastWateredTimer(None)

#cannot yet code, code to expore google trained model as i need to buy another plant
#release water when needed
#only if google model says it needs to be watered
#dont need to worry about if photo taken too often as wont take a photo unless been three hours
#dont need to worry about already watered as it wont run into google model, unless been three hours until photo taken and been not been watered recently

import time
import RPi.GPIO as GPIO

# Set up GPIO for the water pump
PUMP_PIN = 17 #change to GPIO pin where pump is connected
GPIO.setmode(GPIO.BCM)
GPIO.setup(PUMP_PIN, GPIO.OUT)

def ReleaseWater(LastWateredTimer):
    """
    Function to release water to the plant based on the conditions (AI model result, time check).
    Updates the last watered time after watering.
    """

    # Get the current time
    CurrentTime=time.time()
    
    # Simulate the condition when AI decides that the plant needs watering
    # If the last watered time is more than 3 hours ago, water the plant
    print("Checking if watering is needed...")
    
    # Checking if more than 3 hours have passed since last watering
    if LastWateredTimer is None or (CurrentTime - LastWateredTimer) >= (3 * 3600):
        print("Watering plant now...")
        GPIO.output(PUMP_PIN, GPIO.HIGH)  # Activate the water pump
        time.sleep(5)                     # Run the pump for 5 seconds (adjust as necessary)
        GPIO.output(PUMP_PIN, GPIO.LOW)   # Turn off the pump
        print("Watering complete.")
        
        # Update the last watered time to the current time
        LastWateredTimer = CurrentTime
        print(f"Last watered time updated to: {time.ctime(LastWateredTimer)}")
    else:
        print("No watering needed yet.")
    
    return LastWateredTimer  # Return the updated last watered time

# Example usage:
LastWateredTimer = None  # Simulating that the plant hasn't been watered yet
LastWateredTimer = ReleaseWater(LastWateredTimer)  # Release water and update timestamp

# Clean up GPIO after use
GPIO.cleanup()

import tensorflow as tf 
import numpy as np
from PIL import Image

def load_labels(labels_path=os.path.join(pathlib.Path(__file__).parent.absolute(), "labels.txt")):
    #loads the labels from text file
    with open(labels_path, "r") as f:
        labels = f.read().splitlines()
        return labels

#load tensor flow lite model
def run_ai_model(image_path,model_path=os.path.join(pathlib.Path(__file__).parent.absolute(), "model.tflite")):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tesnors()

    #get model input and ouput details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #load and preprocess image
    img= Image.open(image_path).convert("RGB").resize((224,224))
    img_array = np.array(img, dtype=np.float32)/255.0 #normalize to 01
    img_array = np.expand_dims(img_array, axis=0)

    #load labels
    lablels = load_labels(labels_path)

    #set input tensor and run inference
    interpreter.set_tensor(input_details[0]['index'],img_array)
    interpreter.invoke()

    #get model output
    output_data = interpreter.get_tenspr(output_details[0]['index'])
    predicted_class = np.argmax(output_data) #gets class with highest probability

    #return predicted label using names
    return labels[predicted_class]

#image_path = "C:\Users\User\OneDrive - Badminton School\Documents\A-LEVELS\A-LEVEL COMPUTER SCIENCE\UNIT 2\NEA PROJECT\NEA coded solution\test_image.jpg.jpg"
image_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "test_image.jpg.jpg") # Note that this allows for relative paths and will work on all operating systems
model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "model.tflite")
labels_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "labels.txt")

#run ai model 
result = run_ai_model(image_path,model_path,labels_path)

if result == "Wilted":
    print("The plant is wilted and needs watering.")
elif result == "Healthy":
    print("The plant is healthy and does not need watering.")