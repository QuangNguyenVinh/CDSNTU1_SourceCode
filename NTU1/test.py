#from SignDetection import signDetect
from Classic import classic
import cv2

cap = cv2.VideoCapture(r'D:\\CDS\\DiRa_CDSNTU1\\PyCDS\\output.avi')
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv2.resize(frame, (320, 240))
    #label, name, area, location, cut = signDetect.detect(frame)
    '''if(label != None):
        print(label)
        print(name)
        print(area)
        cv2.rectangle(frame, (location[0] + cut, location[1]), (location[2], location[3]), (0,255,0), 2, 1)'''
    classic.update(frame)

    cv2.imshow("",frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break