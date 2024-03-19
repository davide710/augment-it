import cv2
from core import rotation, change_brightness_contrast

rotation(['image00003.jpeg'])
im = cv2.imread('image00003.jpeg')
cv2.waitKey(0)
