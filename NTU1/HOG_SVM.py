import cv2
import numpy as np
class HOG_SVM:
    def __init__(self, svmModel, svmModel2):
        self.MIN_AREA = 0.015
        # self.lowerBound = np.array([80,100,100])
        # self.upperBound = np.array([120,255,255])
        self.lowerBound = np.array([73,146,60])
        self.upperBound = np.array([176,255,255])

        self.svm = cv2.ml.SVM_load(svmModel)
        self.svm2 = cv2.ml.SVM_load(svmModel2)
        self.sign = None
        width = height = 32
        self.hog = cv2.HOGDescriptor(_winSize = (width, height),
                    _blockSize = (width // 2, height // 2),
                    _blockStride = (width // 4, height // 4),
                    _cellSize = (width // 4, height // 4),
                    _nbins = 9)

    def classify_image(self, _image, svmModel):
        descriptor = self.hog.compute(_image)
        return int(svmModel.predict(np.array([descriptor]))[1][0])

    def detectHOG_SVM(self, img):
        dst = img.copy()
        img_h, img_w, _ = img.shape
        areaImg = img_h * img_w
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(imgHSV, self.lowerBound, self.upperBound)
        cv2.imshow("mask", mask)
        _,conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(conts) != 0:
            n = len(conts)
            for index in range(0, n):
                x, y, w, h = cv2.boundingRect(conts[index])
                s = w*h
                if(s * 1.0 / areaImg > self.MIN_AREA):
                    crop = img[max(0, y - 5):min(y + h + 5, img_h), max(0, x - 5): min(x + w + 5, img_w)]
                    crop = cv2.resize(crop, (32, 32))
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    flag = self.classify_image(crop, self.svm)
                    flagStop = self.SVM2Layer(flag, crop)
                    if(flag != 0):
                        self.sign = img[max(0, y - 5):min(y + h + 5, img_h), max(0, x - 5): min(x + w + 5, img_w)]
                        return flag, flagStop, (x, y, h, w)
        return 0,0, (0,0,0,0)

    def SVM2Layer(self, flagSign, img):
        if flagSign == 6: #Stop sign
            stopSign = self.classify_image(img, self.svm2)
            if stopSign == 3: #Also stop sign
                return 3
        return 0


hog_svm = HOG_SVM('model/svm_6.xml', 'model/svm_4.xml')
print("HOG WITH SVM READY")


