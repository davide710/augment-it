import cv2
import os
import numpy as np


def change_brightness_contrast(file_list, min_alpha=0.5, max_alpha=1.5, min_beta=-60, max_beta=60, make_annotations=True):
    dir = os.path.join(os.getcwd(), 'br_and_con')
    if not os.path.exists(dir):
        os.mkdir(dir)

    for impath in file_list:
        img = cv2.imread(impath)
        alpha = np.random.uniform(min_alpha, max_alpha)
        beta = np.random.uniform(min_beta, max_beta)
        bc = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        cv2.imwrite(f"br_and_con/{impath.split('/')[-1].split('.')[0]}_bc.jpg", bc)
        
        if make_annotations:
            ann_file = f"{impath.split('.')[0]}.txt"
            new_ann_file = f"br_and_con/{impath.split('/')[-1].split('.')[0]}_bc.txt"
            with open(ann_file, 'r') as old:
                    with open(new_ann_file, 'w') as new:
                        content = old.read()
                        new.write(content)                
