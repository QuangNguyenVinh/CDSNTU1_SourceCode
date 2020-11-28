import cv2
import numpy as np

DST_FOLDER = r'D:\\OpenSource\\my_mask\\'
SRC_FOLDER = r'D:\\OpenSource\\my_mask\\'
id_lane = [34, 34, 34]
white_low = (0,0,180)
white_up = (179,255,255)
for index in range(2020,4041):
    img = cv2.imread(SRC_FOLDER + str(index) + "_color_mask.png")
    #print(index)
    if type(img) is not np.ndarray:
        print(index)
    #blank_image = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #bin = cv2.inRange(hsv, white_low, white_up)
    #blank_image[np.where(bin == 255)[0], np.where(bin == 255)[1]] = id_lane
    #blank_image[:360, :] = [0, 0, 0]
    #blank_image[379:,:] = [0,0,0]
    #cv2.imwrite(DST_FOLDER + str(index) + "_mask.png", blank_image)
    #print("STT: " + str(index))
    #cv2.waitKey(5)
print("DONE")

