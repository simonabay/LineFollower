from easygopigo3 import EasyGoPiGo3
import time
import easygopigo3
import cv2
import numpy as np
from picamera import PiCamera

gpg = EasyGoPiGo3()
#c=PiCamera()


gpg.forward()
# time.sleep(1)
# gpg.turn_degrees(5) #
# gpg.forward()
time.sleep(0.1)
gpg.stop()
# 
# 
# def captureImage():
#     c.capture('/tmp/picture.jpg', use_video_port = True)
#     img = cv2.imread('/tmp/picture.jpg')
#     return img
#             
# # def imgPreprocessing(img):
# #     #YUV-color,Filtering,resize, reshape
# #     #img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
# #     #img = cv2.GaussianBlur(img,  (3, 3), 0)
# #     img = cv2.resize(img, (200, 66))
# #     img=img.reshape(1,66,200,3)
# #     img = img/255
# #     #img=img.astype(np.float32)
# #     return img
# 
# WHEEL_SPEED_CONSTANT=10
# left_turn = (0.4, 1.0)
# right_turn = (1.0, 0.4)
# 
# gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT)
# gpg.set_motor_power(gpg.MOTOR_RIGHT,1.32* WHEEL_SPEED_CONSTANT)
# time.sleep(0.5)
# #gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT * right_turn[0])
# #gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT * right_turn[1])
# #
# gpg.stop()
# 
# img=captureImage()
# cv2.imwrite('/home/pi/Desktop/Data/ss.jpg', img)


# Test Motor

