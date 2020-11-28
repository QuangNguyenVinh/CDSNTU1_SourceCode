import sys
import cv2
import numpy as np
import os

SRC = r"D:\\OpenSource\\DoAnTotNghiep\\M_SignDatasets\\1\\"
path = os.chdir(SRC)
for file in os.listdir(path):
    img = cv2.imread(SRC + file, 1)
    img = cv2.resize(img, (64, 64))
    cv2.imwrite(SRC + file, img)
