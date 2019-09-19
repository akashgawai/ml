import cv2
import sys
import numpy as np
from model import EMR


cv2.ocl.setUseOpenCL(False)

EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']


network = EMR()
network.build_network()

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
feelings_faces = []


for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

while True:
    
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        
        for face in faces:
            (x,y,w,h) = face
            frame = cv2.rectangle(frame,(x,y-30),(x+w,y+h+10),(255,0,0),2)
            newimg = cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
            newimg = cv2.resize(newimg, (48,48), interpolation = cv2.INTER_CUBIC) / 255.
            result = network.predict(newimg)
            if result is not None:
                maxindex = np.argmax(result[0])
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,EMOTIONS[maxindex],(x+5,y-35), font, 2,(255,255,255),2,cv2.LINE_AA) 

    cv2.imshow('Video', cv2.resize(frame,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()