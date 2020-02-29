import cv2
import numpy as np
from SignClassification import signClassify
import time


class SignDetection:
    def __init__(self, model):
        #For Red Sign
        self.lowerRedBound = ()
        self.upperRedBound = ()
        #For Blue Sign
        self.lowerBlueBound = ()
        self.upperbound = ()
        #For both Red and Blue
        self.lowerBound = (73, 146,  60)
        self.upperBound = (176, 255, 255)
        #
        self.minArea = 0.015
        self.sign = model

    def over_lap_area(self, r1, r2):
        x1_left, y1_top, w1, h1 = r1
        x1_right = x1_left + w1
        y1_bottom = y1_top + h1
        x2_left, y2_top, w2, h2 = r2
        x2_right = x2_left + w2
        y2_bottom = y2_top + h2

        s = max(0, (min(x1_right+1, x2_right+1) - max(x1_left-1, x2_left-1)))*max(0,(min(y1_bottom-1, y2_bottom-1) - max(y1_top+1, y2_top+1)))

        return s

    def merge_rects(self, r1, r2):
        x1_left, y1_top, w1, h1 = r1
        x1_right = x1_left + w1
        y1_bottom = y1_top + h1
        x2_left, y2_top, w2, h2 = r2
        x2_right = x2_left + w2
        y2_bottom = y2_top + h2

        return min(x1_left, x2_left), min(y1_top, y2_top), max(x1_right, x2_right) - min(x1_left, x2_left), max(y1_bottom, y2_bottom) - min(y1_top, y2_top)
    def optimize(self, rects):
        while True:
            changed = 0
            for i in range(len(rects)):
                for j in range(i+1, len(rects)):
                    r1 = rects[i]
                    r2 = rects[j]
                    if self.over_lap_area(r1, r2) > 0:
                        rects.append(self.merge_rects(r1, r2))
                        rects.remove(r1)
                        rects.remove(r2)
                        changed = 1
                        break
                if changed == 1:
                    break
            if changed == 0:
                break
        return rects

    def detect(self, frame):
        cut = frame.shape[1] // 2
        frame = frame[:, cut:]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lowerBound, self.upperBound)
        cv2.imshow("mask", mask)
        res = cv2.bitwise_and(frame, frame, mask = mask)
        cv2.imshow("res", res)

        _, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        arr = []

        frameArea = frame.shape[0] * frame.shape[1]

        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            ratio = w/float(h)
            ratioArea = (w*h)/float(frameArea)
            if ratio > 0.8 and ratio < 1.3 and w > 10 and w < 100 and ratioArea >= self.minArea:
                arr.append((x,y,x+w,y+h))


        arr = self.optimize(arr)

        pred_res = []

        s = []

        if len(arr) != 0:
            for a in arr:
                delt = time.time()
                detect = gray[a[1] : a[3], a[0] : a[2]]
                print("Show: " ,time.time() - delt)
                cv2.imshow("gray", detect)
                t = time.time()
                pred = self.sign.getLabel(detect)
                print("Time: ", time.time() - t)
                if(pred != None):
                    location = a
                    s_max = (a[2] - a[0]) * (a[3] - a[1])
                    return pred, "0", s_max, location, cut


                #pred_res.append(pred)
                #s.append((a[2] - a[0]) * (a[3] - a[1]))

            #location =  arr[np.argmax(s)]
            #s_max = np.max(s)
            #max_predict = pred_res[np.argmax(s)]

            #return max_predict, self.sign.label.get(max_predict), s_max, location, cut

        return None, "None" , 0, None, 0 #Label, Name, Area, Location(Rect)





signDetect = SignDetection(signClassify)
print("SIGN DETECTION READY")





