import cv2
video=cv2.VideoCapture(r"faceDetection.mp4")
faceCascade=cv2.CascadeClassifier(r"/home/aiktc/Desktop/image proces/classifiers/haarcascade_frontalface_default.xml")
print(type(video))
while(True):    
    check,frame =video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,
    scaleFactor=1.25,
    minNeighbors=5,
    )
    for(x,y,w,h)in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+w),(0,255,0),3)
    cv2.imshow("video ka first frame",frame)
    key=cv2.waitKey(10)
    if(key == ord('s')):
        break
cv2.destroyAllWindows()
video.release()