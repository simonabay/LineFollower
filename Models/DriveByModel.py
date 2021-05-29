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



def captureImage():
    c.capture('/tmp/picture.jpg', use_video_port = True)
    img = cv2.imread('/tmp/picture.jpg')
    return img
            
def imgPreprocessing(img):
    #YUV-color,Filtering,resize, reshape
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    #img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66),interpolation = cv2.INTER_AREA) 
    img=img.reshape(1,66,200,3)
    img = img/255
    img=img.astype(np.float32)
    return img

def TurnDegrees(degrees, speed):
    # get the starting position of each motor
    StartPositionLeft      = GPG2.get_motor_encoder(GPG2.MOTOR_LEFT)
    StartPositionRight     = GPG2.get_motor_encoder(GPG2.MOTOR_RIGHT)
    
    # the distance in mm that each wheel needs to travel
    WheelTravelDistance    = ((GPG2.WHEEL_BASE_CIRCUMFERENCE * degrees) / 360)
    
    # the number of degrees each wheel needs to turn
    WheelTurnDegrees       = ((WheelTravelDistance / GPG2.WHEEL_CIRCUMFERENCE) * 360)
    
    # Limit the speed
    GPG2.set_motor_limits(GPG2.MOTOR_LEFT + GPG2.MOTOR_RIGHT, dps = speed)
    
    # Set each motor target
    GPG2.set_motor_position(GPG2.MOTOR_LEFT, (StartPositionLeft + WheelTurnDegrees))
    GPG2.set_motor_position(GPG2.MOTOR_RIGHT, (StartPositionRight - WheelTurnDegrees))
    

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
#left_turn = (0.2, 0.8)
#right_turn = (0.2, 0.8)


gpg = easygopigo3.EasyGoPiGo3()

c = PiCamera()
#dataset_path = "/home/pi/group-f/test-sk-learn/TestData"

tempvar = 10

# Friendly names for the direction predictions
#directions = ["Left", "Straight", "Right", "wrong" ]


def captureImage():
    c.capture('/tmp/picture.jpg', use_video_port = True)
    img = cv2.imread('/tmp/picture.jpg')
    #YUV-color,Filtering,resize, reshape
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    #img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66),interpolation = cv2.INTER_AREA) 
    img=img.reshape(1,66,200,3)
    img = img/255
    img=img.astype(np.float32)
    return img

def predict(K,img):
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions = np.argmax((interpreter.get_tensor(output_index)))
    steeringCommand=predictions-steering
    #TurnDegrees(steeringCommand, 20)
    gpg.turn_degrees(K*steeringCommand)  #turning
    gpg.forward()
    time.sleep(1)
    return steering=predictions
        


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
    
    steering=0
    gpg.set_speed(100)


    while True:
        #gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT)
        #gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT)
        
        
        s = time.time()
        img=imgPreprocessing(captureImage())
        #img=imgPreprocessing(img)
        #print(img.shape)
        #plt.imshow(img[0])
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        predictions = np.argmax((interpreter.get_tensor(output_index)))
        print(predictions)
        steeringCommand=predictions-steering
        #TurnDegrees(steeringCommand, 20)
        K=10
        gpg.turn_degrees(K*steeringCommand)  #turning
        gpg.forward()
        time.sleep(0.5)
        steering=predictions

#         if img is not None:             
#             direction = np.argmax(predictions[0])
#                 # based on the prediction, change the wheel power to make turns
#             if direction == 3:
#                 direction = tempvar
# 
#             if direction == 0: # left
#                 gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT * left_turn[0])
#                 gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT * left_turn[1])
#                 tempvar = direction
#             elif direction == 1: #straight
#                 gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT)
#                 gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT)
#                 tempvar = direction
#             elif direction == 2: # right
#                 gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT * right_turn[0])
#                 gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT * right_turn[1])
#                 tempvar = direction


#             else:
#                 print(f"Unknown direction: {direction}")

        e = time.time()
        print(f"Loop Time: {(e-s)} seconds")

    gpg.reset_all()
    gpg.stop()
    
def exit_handler():
    gpg.stop()
    print("Stopped and exit")
atexit.register(exit_handler)


if __name__==_'main'__:
    time.sleep(2)
    #signal.signal(signal.SIGINT, signal_handler)
    # Load TFLite model and allocate tensors.

    interpreter = tflite.Interpreter(model_path='./model.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    
    



