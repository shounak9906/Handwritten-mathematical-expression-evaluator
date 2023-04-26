# Importing the required libraries
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from imutils.contours import sort_contours
import imutils
import keras

# Labelling the characters as numbers for ease of use
class_indices = {'%': 0, '*': 1, '+': 2, '-': 3, '0': 4, '1': 5, '2': 6, '3': 7, '4': 8, '5': 9, '6': 10, '7': 11, '8': 12, '9': 13, '[': 14, ']': 15}

# Importing the previously created digit recognition model
model = keras.models.load_model('/Users/shounakr/Desktop/Final project/model_v2.h5')

def prediction(image):
    # Converts image to integer form
    plt.imshow(image, cmap = 'gray')
    #resizes image to 40x40 pixel form so that it can be recognised and used by the model
    image = cv2.resize(image,(40, 40))
    # Normalise the image to increase contrast between white/grey/black pixels
    normalized_image = cv2.normalize(image, 0, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    # Convert normalized image to a 40x40x1 3D array
    normalized_image = normalized_image.reshape((normalized_image.shape[0], normalized_image.shape[1], 1))
    # Converts normalised image to a 4D array so it is readable by the model
    case = np.asarray([normalized_image])
    # Makes the prediction by finding the most likely value for the set of pixels
    prediction = np.argmax(model.predict(case), axis = -1)
    return ([i for i in class_indices if class_indices[i]==(prediction[0])][0], prediction)

# Read pixels from image file and convert the input into a usable grayscale format
image = cv2.imread('/Users/shounakr/Desktop/Final project/test2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Reduces noise and smoothens the edges of the numbers/symbols in the image
edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 150)
# Find the series of edges that form the contour of each character and stores them
contours = sort_contours(imutils.grab_contours(cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)), method = "left-to-right")[0]

chars = []
#Loopng through each value in countours
for edge in contours:
    # Computes the bounding box of the number/symbol
    (x, y, w, h) = cv2.boundingRect(edge)
    # Filter out bounding boxes, ensuring they are neither too small nor too large
    if w * h > 1200 :
        # Extract the character and make it appear as white foreground on a black background
        # Then grab the width and height of the thresholded image
        roi = gray[y:y + h, x:x + w]
        # Calls the "Prediction" function to predict the character and stores it in a list
        chars.append(prediction(roi))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Reads the characters from   the list and performs the
equation = [i[0] for i in chars]

if any(i in equation for i in ['%', '-', '+', '*']) :
    if '%' in equation :
        print((''.join([i[0] for i in chars]).replace('%', '/')) + " = " + str(eval((''.join(equation)).replace('%', '/'))))

    else :
        print(''.join([i[0] for i in chars]) + " = " + str(eval(''.join(equation))))

else :
    print(''.join(i for i in equation))