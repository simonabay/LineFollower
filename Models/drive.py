#drive Gopigo
import picamera
import time
import numpy as np
import matplotlib.pyplot as plt 

import tflite_runtime.interpreter as tflite
import cv2
import matplotlib.image as mpimg

import argparse
import logging
import signal
import socket
import sys

import random
from pathlib import Path
import string
#from imutils.video import VideoStream

import easygopigo3
from picamera import PiCamera

import imutils as imutils
from joblib import load
import atexit

dataset_path = "/home/pi/Desktop/Data/Image"

def captureImage():
    c.capture('/tmp/picture.jpg', use_video_port = True)
    img = cv2.imread('/tmp/picture.jpg')
    p = Path(f"{dataset_path}")
    p.mkdir(exist_ok=True, parents=True)
    filename = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    timenow = time.strftime("%H:%M:%S:")
    full_filename = f"{dataset_path}/{timenow}_{filename}.jpg"
    cv2.imwrite(full_filename, img)
    print(full_filename) 
    return img
            
def imgPreprocessing(img):
    #YUV-color,Filtering,resize, reshape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    #img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img=img.reshape(1,66,200,3)
    img = img/255
    img=img.astype(np.float32)
    return img


def signal_handler(sig, frame):
    """
    Exit gracefully
    :param sig:
    :type sig:
    :param frame:
    :type frame:
    :return:
    :rtype:
    """
    print("You pressed ctrl-c resetting GPG and exiting")
    gpg.reset_all()
    sys.exit()
    
# Set the speed.  This is a value that can be experimented with.  You have to
# maintain the right balance between speed, turning rate and inference speed.
WHEEL_SPEED_CONSTANT = 30
left_turn = (0.4, 1.0)
right_turn = (1.0, 0.4)



gpg = easygopigo3.EasyGoPiGo3()

gpg.set_speed(50)

c = PiCamera()
#dataset_path = "/home/pi/group-f/test-sk-learn/TestData"

tempvar = 10

# Friendly names for the direction predictions
directions = ["Left", "Straight", "Right", "wrong" ]


if __name__ == '__main__':
    
    time.sleep(2)
    #signal.signal(signal.SIGINT, signal_handler)
    # Load TFLite model and allocate tensors.

    interpreter = tflite.Interpreter(model_path='./model.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

   
    loop_count = 0
    
    time.sleep(3) # just give the system time to settle out
    print('Starting')
    
    while True:
        s = time.time()
        img=imgPreprocessing(captureImage())
        #time.sleep(0.5)
        #img=imgPreprocessing(img)
        print(img.shape)
        #plt.imshow(img[0])
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        predictions = np.argmax(interpreter.get_tensor(output_index))
        print(predictions)
        if img is not None:             
            direction = predictions
            print(direction)
                # based on the prediction, change the wheel power to make turns
            if direction == 3:
                direction = tempvar

            if direction == 2: # left
                gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT * left_turn[0])
                gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT * left_turn[1])
                time.sleep(0.5)
                #gpg.turn_degrees(3)
                tempvar = direction
            elif direction == 0: #straigh
                gpg.forward()
                time.sleep(0.5)
                #gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT)
                #gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT)
                #tempvar = direction
            elif direction == 1: # right
                gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT * right_turn[0])
                gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT * right_turn[1])
                tempvar = direction
                time.sleep(0.5)

            else:
                print(f"Unknown direction: {direction}")

        e = time.time()
        print(f"Loop Time: {(e-s)} seconds")
        
    gpg.reset_all()
    gpg.stop()
    
def exit_handler():
    gpg.stop()
    print("Stopped and exit")
atexit.register(exit_handler)




