import csv
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
import gopigo3


gpg = EasyGoPiGo3()
c = PiCamera()
GPG2 = gopigo3.GoPiGo3()


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

global turn
# global steerings

# global image_names
# image_names=[]


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
    


def loop(cond):
    global steering
    global steerings
    steerings=[]
    steering=0
    gpg.set_speed(80)
    
    while True:
        logging.debug('Starting producer thread')
        # Keyboard character retrieval method is called and saved
        # into variable
        #gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT)
        #gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT)
        print('test')
        print(steering)
        char = getch()
        # The car will drive forward when the "w" key is pressed
        if(char == "w"):
            #gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT)
            #gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT)
            #gpg.turn_degrees(steering)
            #steering=0
            #gpg.turn_degrees(steering)
            gpg.forward()
            time.sleep(0.5)
            steerings.append(steering)
            print(steering)
            #steerings.append(steering)
            #turn = 'Forward'
            
            #camera("Forward")

        # The car will reverse when the "s" key is pressed
        if(char == "s"):
            gpg.stop()
            #turn = None
            #steerings.append(steering)
            #camera("0")


        # The "a" key will toggle the steering left
        if(char == "a"):
            #turn = 'Left'
            #gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT * left_turn[0])
            #gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT * left_turn[1])
           
            #TurnDegrees(-2, 20)
            #time.sleep(0.5)
            gpg.turn_degrees(-10)
            gpg.forward()
            time.sleep(0.5)
            print(steering)
            steering=steering-10
            steerings.append(steering)
            #camera("Left")

        # The "d" key will toggle the steering right
        if(char == "d"):
            #turn = 'Right'
            
            #gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT * right_turn[0])
            #gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT * right_turn[1])
            
            #TurnDegrees(40, 20)
            gpg.turn_degrees(10)
            gpg.forward()
            time.sleep(0.5)
            steering =steering + 10
            steerings.append(steering)
            print(steering)
            #camera("Right")


        # The "l" key will toggle the LEDs on/off
        if(char == "l"):
            turn = 'wrong'

        # The "x" key will break the loop and exit the program
        if(char == "x"):
            #turn = None
            gpg.stop()
            print("Program Ended")
            break

       
        with cond:
            logging.debug('Making resource available')
            cond.notifyAll()

        # The keyboard character variable will be set to blank, ready
        # to save the next key that is pressed
        char = ""
        
    with open('/home/pi/Desktop/Data/imSteer.csv', 'a') as f:
        #f.writelines("%s\n" % i for i in image_names)
        writer=csv.writer(f)
        for j in range(len(image_names)):
                       writer.writerow([image_names[j],round(steerings[j],4)])
                       
                    
def camera(cond):
    global image_names
    image_names=[] 
    while 1:
        print('test2')
        with cond:
            logging.debug('Making resource available')
            cond.wait()
        #direction = turn
        #if direction is not '0':
        c.capture('/tmp/picture.jpg')
        img = cv2.imread('/tmp/picture.jpg')
        
            
            #dim = (int(320),int(240))
            #dim = (H*0.5,W*0.5)
            #img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            
        #(H, W, C) = img.shape
            #roi = img[int(H*0.8):int(H*1), int(0):int(W)]
            #roi = img[210:270, 264:456]
            
        #thresh = cv2.cvtColor(img, cv2.COLOR_BGR2)
        #(T, thresh) = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            
        p = Path(f"{dataset_path}")
        p.mkdir(exist_ok=True, parents=True)
        filename = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
        timenow = time.strftime("%H:%M:%S:")
        full_filename = f"{dataset_path}/{timenow}_{filename}.jpg"
        cv2.imwrite(full_filename, img)
        print(full_filename)
        image_names.append(full_filename)
        
#         l=[str(steering)]
#         with open('some.csv', 'a') as f:
#             #writer = csv.writer(f)
#             f.writelines("%s\n" % i for i in l)
     

condition = threading.Condition()
t = threading.Thread(name='camera', target=camera, args=(condition,))
w = threading.Thread(name='loop', target=loop, args=(condition,))

if __name__ == "__main__":
    w.start()
    t.start()
    
  

    


