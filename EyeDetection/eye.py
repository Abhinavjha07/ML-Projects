from eyetracker import EyeTracker
import imutils
import argparse
import cv2

p=argparse.ArgumentParser()
p.add_argument("-f","--face",required=True,help="path of face cascade")
p.add_argument("-e","--eye",required=True,help="path of eye cascade")
p.add_argument("-v","--video",help="path of video")

args=vars(p.parse_args())

et=EyeTracker(args["face"],args["eye"])

if not args.get("video",False):
    cam=cv2.VideoCapture(0)
else:
    cam=cv2.VideoCapture(args["video"])

while True:
    (grabbed,frame)=cam.read()
    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame,width=300)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects=et.track(gray)

    for rect in rects:
        cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),2)
    cv2.imshow("Tracking",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
    
