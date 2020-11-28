import cv2
import fnmatch
import os
import numpy as np
src_folder = r'D:\\OpenSource\\DoAnTotNghiep\\H_SignDatasets\\1\\'
dst_folder = r'D:\\OpenSource\\DoAnTotNghiep\\H_SignDatasets\\64x64\\1\\'
image_list = os.listdir(src_folder)
n = len(fnmatch.filter(os.listdir(src_folder), '*.jpg'))
name_sign = r'sign_'
alphaVal = 0.7
betalValDist = [0.5,0.6]
width = height = 64
#Change constrast image
def random_gamma(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha, beta)

#Change shape image (rotation, shear,...)
def affine(img, delta_pix):
    rows, cols = img.shape[:2]
    pts1 = np.float32([[0,0], [rows,0], [0, cols]])
    pts2 = pts1 + delta_pix
    M = cv2.getAffineTransform(pts1, pts2)
    res = cv2.warpAffine(img, M, (rows, cols))
    return res
def affine_dir(img, max_delta_pix):
    delta_pix = np.float32(np.random.randint(-max_delta_pix,max_delta_pix+1,[3,2]))
    img_a = affine(img, delta_pix)
    return img_a


for image in image_list:
    img = cv2.imread(src_folder + image, 1) #1 = RGB image
    img = cv2.resize(img, (width, height))

    for l in range(2):
        for index in range(0, 3):
            new = affine_dir(img, index + l)
            n += 1
            cv2.imwrite(dst_folder + name_sign + str(n) + ".jpg", new)
            for betaVal in betalValDist:
                new2 = random_gamma(new.copy(), alphaVal, betaVal)
                n += 1
                cv2.imwrite(dst_folder + name_sign + str(n) + ".jpg", new2)

print("Done")




