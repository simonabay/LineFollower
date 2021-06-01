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
import threading
import logging

import easygopigo3
from picamera import PiCamera

import imutils as imutils
from joblib import load
import atexit


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
    

WHEEL_SPEED_CONSTANT = 30

gpg = easygopigo3.EasyGoPiGo3()

c = PiCamera()
#dataset_path = "/home/pi/group-f/test-sk-learn/TestData"

tempvar = 10

# Friendly names for the direction predictions
#directions = ["Left", "Straight", "Right", "wrong" ]
#Image Processing

def img_preprocess(img):
  img = img[200:,:,:]
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img,  (3, 3), 0)
  img = cv2.resize(img, (200, 66))
  img = img/255
  img=img.reshape(1,66,200,3)
  img=img.astype(np.float32)
  return img


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        self.error = self.set_point - measurement # proportional error
        self.integral += self.error   # integral error
        return self.Kp * self.error + self.Ki * self.integral

contoller = SimplePIController(0.1, 0.002)


def camera(cond):
    while 1:
        print('test2')
        with cond:
            logging.debug('Making resource available')
            cond.wait()
        c.capture('/tmp/picture.jpg', use_video_port = True)
        img = cv2.imread('/tmp/picture.jpg')
        return img_preprocess(img)
     
def loop(cond,img):
    while True:
        logging.debug('Starting producer thread')
        
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        steering_angle= interpreter.get_tensor(output_index)
        command = contoller.update(steering_angle)
        contoller.set_desired(steering_angle)
        #TurnDegrees(steeringCommand, 20)
        gpg.turn_degrees(command)  #turning
        gpg.forward()
        time.sleep(0.5)
        
        with cond:
            logging.debug('Making resource available')
            cond.notifyAll()
    


condition = threading.Condition()
t = threading.Thread(name='camera', target=camera, args=(condition,))
w = threading.Thread(name='loop', target=loop, args=(condition,img))

if __name__ == "__main__":
   
    time.sleep(2)

    interpreter = tflite.Interpreter(model_path='./model.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

       
    loop_count = 0
    
    time.sleep(3) # just give the system time to settle out
    print('Starting')
    
    gpg.set_speed(100)
    
    w.start()
    t.start()

gpg.reset_all()
gpg.stop()
    
def exit_handler():
    gpg.stop()
    print("Stopped and exit")
atexit.register(exit_handler)


