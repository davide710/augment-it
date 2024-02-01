import cv2
import numpy as np
import os


def _resize(img): #for debug purposes
    width = 400
    scale_factor = width / img.shape[1]
    height = int(img.shape[0] * scale_factor)
    dimension = (width, height)
    im = cv2.resize(img, dimension, interpolation = cv2.INTER_AREA)
    return im


def rotation(file_list, min_angle=5, max_angle=355, make_annotations=True):
    for name in file_list:
        img = cv2.imread(f"{name}")
        rows, cols = img.shape[:2]

        max_dim = int(np.sqrt(cols**2 + rows**2))
        background = np.zeros((max_dim, max_dim, 3))
        upper_left = ((max_dim - rows) // 2, (max_dim - cols) // 2)

        for i in range(upper_left[0], upper_left[0] + rows):
            for j in range(upper_left[1], upper_left[1] + cols):
                for k in range(3):
                    background[i, j, k] = img[i - upper_left[0]][j - upper_left[1]][k] #/ 255 for imshow

        angle = np.random.uniform(min_angle, max_angle)
        O = (max_dim/2, max_dim/2)
        matrix = cv2.getRotationMatrix2D(O, angle, 1)
        rotated = cv2.warpAffine(background, matrix, (max_dim, max_dim))
        rescaled = cv2.resize(img, (cols, rows), interpolation = cv2.INTER_AREA)

        if make_annotations:
            ann_file = f"{name.split('.')[0]}.txt"
            classes = []
            boxes = []
            with open(ann_file, 'r') as f:
                for line in f.readlines():
                    classes.append(line.split()[0])
                    boxes.append(line.split()[1:])
            
            new_ann_file = f"{name.split('.')[0]}_rot.txt"
            
                
                    




