import time
import tensorflow as tf
from keras.models import load_model # type: ignore
import numpy as np
from PIL import Image, ImageOps
import os, pathlib, h5py
import testmodel
import turn_pump_on


def last_photo_taken_timer(photo_taken_time):
    current_time = time.time()
    if photo_taken_time is not None:
        timer = current_time - photo_taken_time
        #print(timer)
        return timer
    else:
        return 0

# Simulate a photo taken 3 hours ago
photo_taken_time = time.time() - (2 * 60 * 60)
#elapsed_time = last_photo_taken_timer(photo_taken_time)

#print(f"Elapsed time since last photo: {elapsed_time} seconds")


def check_photo_timer(photo_taken_time):
    if last_photo_taken_timer(photo_taken_time) >= 10800:
        def WOW():
            pass#print("call wow")#wow fucntion called
        WOW()
        print("true")
        return True
    else:
        print("no photo taken")
        return False

if check_photo_timer(photo_taken_time) :
    def turn_pump_on():
        turn_pump_on()
    
#10800s is three hours
#if time since last photo is more than three hours then take a photo
# call teachable machine model to predict the image
#call turn pump on fucntion 