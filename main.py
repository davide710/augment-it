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

def _draw(img, box): #testing
    cv2.circle(img, (int(box[0] * img.shape[1]), int(box[1] * img.shape[0])), 10, (255, 0, 0), -1)

def rotation(file_list, min_angle=5, max_angle=355, make_annotations=True):
    for name in file_list:
        img = cv2.imread(f"{name}")
        rows, cols = img.shape[:2]

#        max_dim = int(np.sqrt(cols**2 + rows**2))
#        background = np.zeros((max_dim, max_dim, 3))
#        upper_left = ((max_dim - rows) // 2, (max_dim - cols) // 2)
#
#        for i in range(upper_left[0], upper_left[0] + rows):
#            for j in range(upper_left[1], upper_left[1] + cols):
#                for k in range(3):
#                    background[i, j, k] = img[i - upper_left[0]][j - upper_left[1]][k] #/ 255 for imshow

        angle = np.random.uniform(min_angle, max_angle)
        O = (cols / 2, rows / 2)
        matrix = cv2.getRotationMatrix2D(O, angle, 1)
        rotated = cv2.warpAffine(img, matrix, (cols, rows))
#        rescaled = cv2.resize(rotated, (cols, rows), interpolation = cv2.INTER_AREA)

        if make_annotations:
            ann_file = f"{name.split('.')[0]}.txt"
            classes = []
            boxes = []
            with open(ann_file, 'r') as f:
                for line in f.readlines():
                    classes.append(line.split()[0])
                    box = [float(i) for i in line.split()[1:]]
                    boxes.append(box)
                    _draw(img, box)
                    cv2.imshow('', img)

            
            new_ann_file = f"{name.split('.')[0]}_rot.txt"
            with open(new_ann_file, 'w') as f:
                for cl, box in zip(classes, boxes):
                    ox, oy, w, h = box
                    C = np.array([-ox, oy]) # x must be negative for calculations because image coordinates have origin in high left and go down and right 
                    O = np.array([- O[0] / cols, O[1] / rows])
                    v = C - O
                    theta = angle * np.pi / 180
                    matrix = matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    v_rot = np.dot(matrix, v)
                    C_rot = v_rot + O
                    new_center = np.array([- C_rot[0], C_rot[1]])
                    # A-----B
                    # |  C  |
                    # E-----D
                    A = C + np.array([w, - h])
                    B = C + np.array([-w, - h])
                    D = C + np.array([-w, h])
                    E = C + np.array([w, h])
                    corners_rot = np.array([np.dot(matrix, P) for P in [A, B, D, E]])
                    x_rot = [P[0] for P in corners_rot]
                    y_rot = [P[1] for P in corners_rot]
                    highest = corners_rot[np.argmax(y_rot)]
                    leftest = corners_rot[np.argmax(x_rot)]
                    new_width = 2 * np.sqrt(np.dot((highest - C_rot), (highest - C_rot))) * np.sin(theta)
                    new_height = 2 * np.sqrt(np.dot((leftest - C_rot), (leftest - C_rot))) * np.cos(theta)
                    _draw(rotated, [new_center[0], new_center[1], new_width, new_height])
                    cv2.imshow('r', rotated)
                    f.write(f'{cl} {new_center[0]} {new_center[1]} {new_width} {new_height}')


                    
rotation(['image00003.jpeg'])
im = cv2.imread('image00003.jpeg')
cv2.waitKey(0)


