import cv2
import os
import random
import time
width = height = 32
random.seed(time.time())

#
src_file = r'D:\\OpenSource\\DataFull\\DataFull1\\'
image_list = os.listdir(src_file)
dst_file = r'D:\\OpenSource\\DoAnTotNghiep\\0\\'

countFile = 0

#Single image
numOfCrop = 5

#Min and max for point coordinate
def randomPoint(maxW, maxH):
    x = random.randrange(0, maxW, width)
    y = random.randrange(0, maxH, height)
    return x,y

def randomPoint2(maxW, maxH):
    x = int(random.random() * (maxW - 2*width) + width) #from width to maxW-width
    y = int(random.random() * (maxH - 2*height) + height) #form height to maxH-height
    return x,y

def crop_1_Img(img):
    #h,w = img.shape[:2]
    w,h = 320,80
    x,y = randomPoint2(w,h)
    print(x,y)
    new_img = img[y:y+height, x:x+width]
    return new_img


def excecuteCtop():
    global countFile
    for image in image_list:
        img = cv2.imread(src_file + image)
        for index in range(numOfCrop):
            new_img = crop_1_Img(img)
            '''try:
                cv2.imshow("img", new_img)
            except Exception:
                print(new_img.shape)
            cv2.waitKey(1)'''
            cv2.imwrite(dst_file + str(countFile) + ".png", new_img)
            countFile += 1
            print(countFile)

    print("Number of image:",countFile)
    print("Done")

excecuteCtop()
print("Number of image:", countFile)
print("Number of image: %d"%countFile)
print("Number of image: {}".format(countFile))



