import cv2
import numpy as np
import os
SRC_FOLDER = r'D:\\OpenSource\\DoAnTotNghiep\\no_turn_left_sign\\'
image_list = os.listdir(SRC_FOLDER)
for image in image_list:
    img = cv2.imread(SRC_FOLDER + image)
    if type(img) is not np.ndarray:
        print(image)

print("DONE")