import cv2
import time
DST_FOLDER = r'D:\\OpenSource\\DoAnTotNghiep\\no_turn_left_sign\\'
SRC_FOLDER = r'D:\\CDS\\OpenCVProjs\\OpenCVProjs\\no_turn_left\\'

TXT_FILE = r'D:\\OpenSource\\DoAnTotNghiep\\brokenfile.txt'

f = open(TXT_FILE, "r")
t = time.time()
for x in f:
    strs = x.split()
    if len(strs) != 6:
        continue
    name = strs[0]
    img = cv2.imread(SRC_FOLDER + name)
    x,y,w,h = list(map(int, strs[2:]))
    print(name,x,y,w,h)
    #cv2.imshow("",img[y:y+h, x:x+w])
    #cv2.waitKey(1)
    cv2.imwrite(DST_FOLDER + name, img[y:y+h, x:x+w])
print("DONE")