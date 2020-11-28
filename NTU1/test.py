#from SignDetection import signDetect
from Classic import classic
from Advance import *
import cv2

cap = cv2.VideoCapture(r'output4.avi')
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv2.resize(frame, (240, 160))
    center = Advance_lane(frame)
    cv2.circle(frame, (center,80),5, (0,0,255))
    print(center)
    cv2.imshow("Frame", frame)

    cv2.imshow("",frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break