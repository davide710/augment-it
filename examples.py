import cv2
from core import rotation, change_brightness_contrast
import os

files_list = os.listdir('resources/images_and_annotations')
paths_list = [f'resources/images_and_annotations/{fname}' for fname in files_list]
images_list = [impath for impath in paths_list if impath.split('.')[-1] != 'txt']

circles = [impath for impath in images_list if int(impath.split('.')[0][-3:]) < 300]
rectangles = [impath for impath in images_list if int(impath.split('.')[0][-3:]) > 300]

#rotation(circles, rotate_bbox=False)
rotation(rectangles, rotate_bbox=True)
