import cv2
import numpy as np

img = cv2.imread('../od/money/org_data/train/image00003.jpeg')

rows, cols, _ = img.shape
#max_dim = int(np.sqrt(cols**2 + rows**2))
#background = np.zeros((max_dim, max_dim, 3))
#upper_left = ((max_dim - rows) // 2, (max_dim - cols) // 2)
#
#for i in range(upper_left[0], upper_left[0] + rows):
#    for j in range(upper_left[1], upper_left[1] + cols):
#        for k in range(3):
#            background[i, j, k] = img[i - upper_left[0]][j - upper_left[1]][k] / 255 #/ 255 for imshow

angle = 45#random.choice(angles)
matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
rotated = cv2.warpAffine(img, matrix, (cols, rows))
rescaled = cv2.resize(rotated, (cols, rows), interpolation=cv2.INTER_AREA)

cv2.imshow('1', img)
cv2.imshow('2', rescaled)
cv2.waitKey(0)
