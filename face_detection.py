#Face Detection Using OpenCV
#Developed a face detection system using OpenCV and Haar Cascade Classifier in Python.
#Processed images by converting them to grayscale and detecting multiple faces using the Haar Cascade technique.
#Drew bounding boxes around detected faces with real-time face counting.
#Demonstrated proficiency in image processing and object detection algorithms.



import cv2 as cv
import numpy as np
img=cv.imread("family.jpeg")
cv.imshow("Image",img)
#change normal image to grayscale image
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Gray person",gray)
#harcascade is more popular and require minimal requirements
#cascadeclassifier is used to detect faces in the image
haar_cascade=cv.CascadeClassifier('hear_face.xml')
faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
#it will return number of faces in the images
print(f'Number of faces found={len(faces_rect)}')
#rectangle an faces in the image,X-axis,y-axis,width,height 255 is green color
for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow('Detected faces',img)
cv.waitKey()