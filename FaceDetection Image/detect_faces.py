from facedetector import FaceDetector
import argparse
import cv2


parser=argparse.ArgumentParser()
parser.add_argument("-f","--face",required=True,help="path of face cascade")
parser.add_argument("-i","--image",required=True,help="path of image")
args=vars(parser.parse_args())

image=cv2.imread(args["image"])
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

fd=FaceDetector(args["face"])

faceRects=fd.detect(gray,scaleFactor=1.2,minNeighbors=5,minSize=(30,30))

print("{} faces".format(len(faceRects)))


for (x,y,w,h) in faceRects:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2) #BGR

cv2.imshow("Faces",image)

cv2.waitKey(0)
