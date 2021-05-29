import sys, tty, termios, time
import time
from pathlib import Path
from easygopigo3 import EasyGoPiGo3
import cv2
from picamera import PiCamera
import random
import string
from matplotlib import pyplot as plt
#from joblib import dump, load
import threading
import logging

gpg = EasyGoPiGo3()
c = PiCamera()


WHEEL_SPEED_CONSTANT = 20


left_turn = (0.4, 1.0)
right_turn = (1.0, 0.4)

dataset_path = "/home/pi/Desktop/Images"

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

print("w/s: acceleration")
print("a/d: steering")
print("l: lights")
print("x: exit")




def loop(cond):
    while True:
        logging.debug('Starting producer thread')
        # Keyboard character retrieval method is called and saved
        # into variable
        print('test')
        
        char = getch()
        global turn
       
        # The car will drive forward when the "w" key is pressed
        if(char == "w"):
            gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT)
            gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT)
            turn = 'Forward'
           
            
            #camera("Forward")

        # The car will reverse when the "s" key is pressed
        if(char == "s"):
            gpg.stop()
            turn = None
            #camera("0")


        # The "a" key will toggle the steering left
        if(char == "a"):
            turn = 'Left'
            gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT * left_turn[0])
            gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT * left_turn[1])
            
            #camera("Left")

        # The "d" key will toggle the steering right
        if(char == "d"):
            turn = 'Right'
            gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT * right_turn[0])
            gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT * right_turn[1])
            
            #camera("Right")


        # The "l" key will toggle the LEDs on/off
        if(char == "l"):
            turn = 'wrong'

        # The "x" key will break the loop and exit the program
        if(char == "x"):
            turn = None
            gpg.stop()
            print("Program Ended")
            break

       
        with cond:
            logging.debug('Making resource available')
            cond.notifyAll()

        # The keyboard character variable will be set to blank, ready
        # to save the next key that is pressed
        char = ""


def camera(cond):
    while 1:
        print('test2')
        with cond:
            logging.debug('Making resource available')
            cond.wait()
        direction = turn
        if direction is not '0':
            c.capture('/tmp/picture.jpg')
            
            img = cv2.imread('/tmp/picture.jpg')
            
            
            #dim = (int(320),int(240))
            #dim = (H*0.5,W*0.5)
            #img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            
            (H, W, C) = img.shape
            #roi = img[int(H*0.8):int(H*1), int(0):int(W)]
            #roi = img[210:270, 264:456]
            
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            (T, thresh) = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            
            p = Path(f"{dataset_path}/{direction}")
            p.mkdir(exist_ok=True, parents=True)
            filename = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
            timenow = time.strftime("%H:%M:%S:")
            full_filename = f"{dataset_path}/{direction}/{timenow}_{filename}_{direction}.jpg"
            cv2.imwrite(full_filename, thresh)
  
condition = threading.Condition()
t = threading.Thread(name='camera', target=camera, args=(condition,))
w = threading.Thread(name='loop', target=loop, args=(condition,))

w.start()
t.start()

