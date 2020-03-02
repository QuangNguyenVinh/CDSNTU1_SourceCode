import cv2
import numpy as np

class Classic:
    def __init__(self):
        self.k_median = 5
        self.kernel_size = 3
        self.thres_val = 100
        self.skyLine = 100
        self.a = 120.0 / 320
        self.b = 30
        self.tolerance = 30
        self.alpha = 0.2
        self.laneRight = []
        self.laneLeft = []
        self.scan = 4
        self.lowWhite = (0,0,180)
        self.upWhite = (179,30,255)

    def whiteLane(self, src):
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        bin =  cv2.inRange(hsv, self.lowWhite, self.upWhite)
        bin2 = bin.copy()
        dist = cv2.distanceTransform(bin, cv2.DIST_L2, 3)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        cv2.imshow("dist", dist)
        _, dist = cv2.threshold(dist,0.1, 1.0, cv2.THRESH_BINARY)
        cv2.imshow("dist2", dist)
        dist_8u = dist.astype('uint8')
        _, contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bin2, contours, -1, 0, -1)
        cv2.imshow("bin2", bin2)
        return contours


    def reduceNoise(self, src):
        dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        return cv2.medianBlur(dst, self.k_median)

    def LaplacianEdgeDetect(self, gray):
        dst = cv2.Laplacian(gray, cv2.CV_32F, ksize = self.kernel_size)
        dst = cv2.convertScaleAbs(dst)
        ret, bin = cv2.threshold(dst, self.thres_val, 255, cv2.THRESH_BINARY)
        cv2.imshow("bin", bin)

        return bin

    def View(self, src):
        h, w = src.shape[:2] #H x W
        view = src[self.skyLine : ,]
        return view

    def ignoreMid(self, x):
        return self.a * x + self.b

    def findLane(self, bin):
        self.laneLeft = []
        self.laneRight = []
        left, right = 0, 0
        midLane = bin.shape[1] // 2 #W/2
        start = bin.shape[0] - 50
        high = start
        while(high > 50):

            i = int(midLane - self.ignoreMid(high))
            while(i > self.scan):
                try:
                    scan = bin[high : high + self.scan, i : i + self.scan]
                except Exception as e:
                    print(e)
                    i = i - self.scan // 2
                    continue


                if (cv2.countNonZero(scan) * 100 > self.scan ** 2 * self.tolerance):
                    self.laneLeft.append((i, high + self.skyLine))
                    left = i
                    break
                left = 0

                i = i - self.scan // 2



            j = int(midLane + self.ignoreMid(high))
            while(j < bin.shape[1] - self.scan):
                try:
                    scan = bin[high : high + self.scan, j : j + self.scan]
                except Exception as e:
                    print(e)
                    j = j + (self.scan // 2)
                    continue
                if(cv2.countNonZero(scan) * 100 > self.scan ** 2 * self.tolerance):
                    self.laneRight.append((j, high + self.skyLine))
                    right = j
                    break
                right = 0

                j = j + (self.scan // 2)


            if (left * right > 0):
                midLane = int(midLane * self.alpha + ((left + right) / 2) * (1 - self.alpha))
                high = high - self.scan
                continue

            if(left + right == 0):
                high = high - self.scan
                continue

            if (left == 0):
                left = int(midLane - self.ignoreMid(high) * 2)
            if (right == 0):
                right = int(midLane + self.ignoreMid(high) * 2)

            midLane = int(midLane * self.alpha + ((left + right) / 2) * (1 - self.alpha))

            high = high - self.scan


    def update(self, src):
        cnts = self.whiteLane(src)
        bin = self.reduceNoise(src).copy()
        bin = self.LaplacianEdgeDetect(bin)
        bin = self.View(bin).copy()
        self.findLane(bin)

    def getLeftLane(self):
        return self.laneLeft

    def getRightLane(self):
        return self.laneRight

classic = Classic()

print("CLASSIC LANE READY")



