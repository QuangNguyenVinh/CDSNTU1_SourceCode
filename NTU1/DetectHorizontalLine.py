import cv2
import numpy as np

class DetectLine:

    def __init__(self):
        print("Detect Horizontal Line!")


    def isHorizontal(self, theta, maxDegreeDifference=6):
        return (np.abs(theta - np.pi / 2) < (np.deg2rad(maxDegreeDifference)))

    def findLine(self, thres, minLineLength):
        lines = cv2.HoughLines(thres, rho=1, theta=np.deg2rad(1), threshold=minLineLength)

        if self.isHorizontal(lines[0, 0, 1]):
            return lines[0, 0, 0], lines[0, 0, 1]

        line = lines[np.argmin(np.abs(lines[:, 0, 1] - np.pi / 2.))]

        rho, theta = line[0, 0], line[0, 1]

        if not self.isHorizontal(theta):
            return None

        return rho, theta

    def detect_parking(self, img, minLineLength=140):

        img = np.uint8(img)
        thres = cv2.Canny(img,0,255)
        cv2.imshow("Thres: ", thres)
        try:
            rho, theta = self.findLine(thres, minLineLength)
            if rho < 0:
                raise Exception
        except:
            return False, (0, 0), (0, 0)

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        pt1 = (x1, y1)
        pt2 = (x2, y2)

        return True, pt1, pt2



line_ = DetectLine()

