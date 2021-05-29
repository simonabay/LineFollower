import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Load TFLite model and allocate tensors.

interpreter = tflite.Interpreter(model_path='./model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

predictions = []
test_labels = []
test_images = []
label=1

#img=cv2.imread('/home/pi/Desktop/Images/imageTest.png')
#img=cv2.imread('/home/pi/Desktop/Images/imm.jpg')
img=cv2.imread('/home/pi/Desktop/Data/r.jpg')
print(img.shape)
plt.imshow(img)

#crop image
height, width, _= img.shape #Image shape
new_width, new_height =(200,66)
#Calculate the target image coordinates
# 
# left=(width-new_width)//2
# top=(height-new_height)//2
# right=(width+new_width)//2
# bottom=(height+new_height)//2
# 
# im=img[left:right,top:bottom,:]
# print(im.shape)

img=cv2.resize(img,(200,66))
img=img.reshape(1,66,200,3)
img=img/255
print(img.shape)
img=img.astype(np.float32)
interpreter.set_tensor(input_index, img)
interpreter.invoke()
predictions.append(interpreter.get_tensor(output_index))
test_labels.append(label)
#   #plt.imshow(img[0])
# test_images.append(np.array(img))
print(np.argmax(predictions))