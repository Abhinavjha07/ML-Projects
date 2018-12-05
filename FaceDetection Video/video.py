from facedetector import FaceDetector
import imutils
import argparse
import cv2

parser=argparse.ArgumentParser()
parser.add_argument("-f","--face",required=True,help="path of face cascade")
parser.add_argument("-v","--video",help="path of video")

args=vars(parser.parse_args())

fd=FaceDetector(args["face"])

if not args.get("video",False):
    vid=cv2.VideoCapture(0)

else:
    vid=cv2.VideoCapture(args["video"])


while True:
    (grabbed,frame)=vid.read()

    if args.get("video") and not grabbed :
        break

    frame=imutils.resize(frame,width=300)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faceRects=fd.detect(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
    frameClone=frame.copy()

    for (x,y,w,h) in faceRects:
        cv2.rectangle(frameClone,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("Face",frameClone)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
