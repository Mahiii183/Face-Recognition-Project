import cv2
import numpy as np
import face_recognition
import os

path = 'ImagesAttendance'
# STEP 6: LOAD IMAGES AND CONVERT THEM TO RGB

# STEP 7: ALL THIS PROCESS IS VERY REPETITIVE AND MESSY SO CREATE A LIST THAT FINDS THE IMAGES FROM OUR
# FOLDER AND GENERATE THE ENCODINGS AUTOMATICALLY AS WELL AND THEN TRY TO FIND IT IN OUR WEBCAM

path = 'ImagesAttendance'
images = [] # LIST OF ALL THE IMAGES
classNames = [] #
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0]) # THIS WILL SAVE THE NAMES WITHOUT THE JPG OR PNG NAMES
print(classNames)

# Step 8: A FUNCTION THAT COMPUTES ALL THE ENCODINGS FOR US
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# STEP 8: COMPARE THE IMAGES WITH A WEBCAM

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25) # One fourth of img size
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceCurrFrame = face_recognition.face_locations(imgS)
    encodesCurrFrame = face_recognition.face_encodings(imgS, faceCurrFrame) # To find encodings of the webcam img

# IN A WEBCAM THERE CAN BE MULTIPLE FACES SO WE NEED TO DETECT THE CORRECT FACE
# STEP 9: WE'LL COMPARE ALL THE FACES ON THE WEBCAM WITH THE ENCODINGS OF THE SAVES IMAGES IN THE LIST TO FIND
# THE REAL PERSON

    for encodeface,faceLoc in zip(encodesCurrFrame,faceCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface) # This will give us 4 values as we have 4 imgs and the one with the lowest dist val is the match
        print(faceDis)
        matchIndex = np.argmin(faceDis) #Gives Index of the lowest dist value

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2), (0,255,0), 3)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

