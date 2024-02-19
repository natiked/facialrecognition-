import cv2
import face_recognition
import os 
import numpy as np
import time
from datetime import datetime

now = datetime.now()
path = input("Specify the location of the image: ")
images = []
classNames = []
myList = os.listdir(path)
now = datetime.now()
current_time = now.strftime("%H:%M:%S")

for image in myList:
    currentImage = cv2.imread(f'{path}/{image}') 
    images.append(currentImage)
    classNames.append(os.path.splitext(image)[0])
print('Students: ' + str(classNames))

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        with open('EncodeList.txt','r+') as f:
             f.writelines(f'{classNames},\n {encode}')
    return encodeList   
KnownEncodings = findEncodings(images)
print("Encoding completed")

cam = cv2.VideoCapture(0)
SleepTime = input("Specify the camera Launch time in seconds: ")
time.sleep(int(SleepTime))
while True:
    ret, img = cam.read()
    imageResize = cv2.resize(img,(0,0),None,0.25,0.25)
    imageResize = cv2.cvtColor(imageResize, cv2.COLOR_BGR2RGB)
    faceFrame = face_recognition.face_locations(imageResize)
    encodedFrame = face_recognition.face_encodings(imageResize,faceFrame)
    for encodeFace,faceLocation in zip(encodedFrame,faceFrame):
        matches = face_recognition.compare_faces(KnownEncodings,encodeFace)
        faceDistance = face_recognition.face_distance(KnownEncodings,encodeFace)
        matchIndex = np.argmin(faceDistance)
        if matches[matchIndex]:
             name = classNames[matchIndex].upper()
             y1,x2,y2,x1 = faceLocation
             y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
             cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
             #                                , Color   ,thickness
             cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
             cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
             
             with open('attendance.txt','r+') as wr:
                 wr.writelines("The Time Of Launch : ")
                 wr.writelines(f'{current_time}') 
                 wr.writelines(f'\n{name}\n')                 
                 with open("attendance.txt", 'r+') as fp:
                     line_numbers = [2, 3]
                     lines = []
                     for i, line in enumerate(fp):
                         if i in line_numbers:
                             lines.append(line.strip())  
                 if name != lines:
                     wr.writelines(f'{name}')                     
    cv2.imshow('Face Recogniton',img)  
    if cv2.waitKey(1) == ord('q'):
         print("Exited")
         break

cam.release()
cv2.destroyAllWindows()

    



