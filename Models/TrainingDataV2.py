import sys, tty, termios, time
from easygopigo3 import EasyGoPiGo3
import time
from picamera import PiCamera
from pathlib import Path
import cv2
import random
import string
from matplotlib import pyplot as plt
from joblib import dump, load


gpg = EasyGoPiGo3()
c = PiCamera()


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

    
def camera2():
    
    c.capture('/tmp/picture.jpg')
    
    img = cv2.imread('/tmp/picture.jpg')
    (H, W, C) = img.shape

    roi = img[int(H*0.7):int(H), int(W*0.25):int(W*0.75)]

    gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    (T, thresh) = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    plt.subplot(111),plt.imshow(thresh,cmap = 'gray')
    plt.title('Pooled Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def loop():
    while True:
#         logging.debug('Starting producer thread')
        # Keyboard character retrieval method is called and saved
        # into variable
        
        char = getch()
        # The car will drive forward when the "w" key is pressed
        if(char == "w"):
            gpg.forward()
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
            gpg.left()
            
            #camera("Left")

        # The "d" key will toggle the steering right
        if(char == "d"):
            turn = 'Right'
            gpg.right()
            
            #camera("Right")

        # The "l" key will toggle the LEDs on/off
        if(char == "l"):
            toggleLights()

        # The "x" key will break the loop and exit the program
        if(char == "x"):
            turn = None
            gpg.stop()
            print("Program Ended")
            break

       
#         with cond:
#             logging.debug('Making resource available')
#             cond.notifyAll()

        # The keyboard character variable will be set to blank, ready
        # to save the next key that is pressed
        char = ""
        return turn


def camera(turn):
    while 1:
        direction = turn
        if direction is not '0':
            c.capture('/tmp/picture.jpg')
            
            img = cv2.imread('/tmp/picture.jpg')
            
            
            dim = (int(320),int(240))
            #dim = (H*0.5,W*0.5)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            
            (H, W, C) = img.shape
            roi = img[int(H*0.75):int(H), int(W*0.2):int(W*0.8)]
            #roi = img[210:270, 264:456]
            
            gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            (T, thresh) = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            
            p = Path(f"{dataset_path}/{direction}")
            p.mkdir(exist_ok=True, parents=True)
            filename = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
            full_filename = f"{dataset_path}/{direction}/{filename}_{direction}.jpg"
            cv2.imwrite(full_filename, thresh)
            
            
if __name__ == "__main__":
    turn=loop()
    time.sleep(2)
    camera(turn)



