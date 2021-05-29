from easygopigo3 import EasyGoPiGo3
import time
import easygopigo3
import cv2
import numpy as np
from picamera import PiCamera

gpg = EasyGoPiGo3()
c=PiCamera()
#gpg.forward()
#time.sleep(1)
#gpg.turn_degrees(5) #

gpg.set_speed(50)
gpg.forward()
time.sleep(2)
gpg.turn_degrees(7) #
gpg.forward()

#gpg.forward() #
time.sleep(1)
gpg.stop()

#while True:
#    gpg.forward()
 #   time.sleep(10)
    #gpg.turn_degrees(20) #
    #time.sleep(1)
    #gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT)
    #gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT)

