#It is based on two papers by Z.Zivkovic, “Improved adaptive Gausian mixture model for background subtraction” in 2004

import cv2
import numpy as np
import queue

cap = cv2.VideoCapture(0)



fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))



bgsMOG = cv2.createBackgroundSubtractorMOG2()


q = queue.Queue(maxsize=50) 

while(1):
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (31, 31), 0)

        fgmask = bgsMOG.apply(gray)
        _,contours, hierarchy = cv2.findContours(fgmask,
                                cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        try:
            hierarchy = hierarchy[0]
        except:
            hierarchy = []

        for contour, hier in zip(contours, hierarchy):

            (x, y, w, h) = cv2.boundingRect(contour)

            if w > 5 and h > 5:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 255, 255), 2)

                if (q.full()):
                    q.get()

                q.put((int(x+w/2), int(y+h/2)))

                '''for elem in list(q.queue):
                    cv2.rectangle(frame, (elem), (elem[0] + 4, elem[1] + 2), (0, 255, 0), 2);'''

        cv2.imshow('Output', frame)

        out.write(frame)

        if cv2.waitKey(30) & 0xff ==ord('q'):
            break
    else: break


cap.release()


cv2.destroyAllWindows()
