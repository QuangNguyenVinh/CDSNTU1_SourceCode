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

    def reduceNoise(self, src):
        dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        return cv2.medianBlur(dst, self.k_median)

    def LaplacianEdgeDetect(self, gray):
        dst = cv2.Laplacian(gray, cv2.CV_32F, ksize = self.kernel_size)
        dst = cv2.convertScaleAbs(dst)
        ret, bin = cv2.threshold(dst, self.thres_val, 255, cv2.THRESH_BINARY)
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
        bin = self.reduceNoise(src).copy()
        bin = self.LaplacianEdgeDetect(bin)
        bin = self.View(bin).copy()
        self.findLane(bin)

    def getLeftLane(self):
        return self.laneLeft

    def getRightLane(self):
        return self.laneRight

classic = Classic()

cap = cv2.VideoCapture(r'D:\\CDS\\DiRa_CDSNTU1\\PyCDS\\output.avi')
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv2.resize(frame, (320, 240))
    classic.update(frame)
    for i in range(0, len(classic.laneLeft)):
        cv2.circle(frame,classic.laneLeft[i], 5, (0,0,255))
    for i in range(0, len(classic.laneRight)):
        cv2.circle(frame, classic.laneRight[i], 5, (0, 255, 0))
    cv2.imshow("",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



