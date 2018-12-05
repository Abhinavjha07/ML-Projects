import cv2
import numpy as np

cap=cv2.VideoCapture(0)

fgbg=cv2.createBackgroundSubtractorMOG2()

while True:
    ret,frame=cap.read()
    
    fgmask=fgbg.apply(frame)
    kernel=np.ones((1,4),np.float32)
    erosion=cv2.erode(fgmask,kernel,iterations=1)
    #dilation=cv2.dilate(fgmask,kernel,iterations=1)
    cv2.imshow('original',frame)
    cv2.imshow('fg',fgmask)
    cv2.imshow('erosion',erosion)
    #cv2.imshow('dilation',dilation)
    
    if cv2.waitKey(30) & 0xff ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
