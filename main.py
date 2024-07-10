import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import mediapipe as mp
import pyttsx3
import threading
import math
import numpy as np

try:
    face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
    classifier = load_model(r'model.h5')
except Exception as e:
    print("error: ",e)

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)
GR_dict={0:(0,255,0),1:(0,0,255)}

hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

MAX_SLEEP_TIME = 60
sleep_timer = MAX_SLEEP_TIME
curr_sign = None

NAME = "User"
print("User name :",NAME)

WIDTH = 640
HEIGHT = 480

def getX(x):
	return x*WIDTH
def getY(y):
	return y*HEIGHT

#GET SIGN'S
def getSign(finger_landmarks):
	fingers = []
	tipIds = [4, 8, 12, 16, 20]
	#Thumb finger	#------NOT WORKING WELL --------
	x1,x2 = getX(finger_landmarks[tipIds[0]].x) , getX(finger_landmarks[tipIds[0] - 1].x)	
	y1,y2 = getX(finger_landmarks[tipIds[0]].y) , getX(finger_landmarks[tipIds[0] - 1].y)
	x3,y3 = getX(finger_landmarks[tipIds[1]-3].x) , getX(finger_landmarks[tipIds[1]-3].y)
	if math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2)) > 10 and math.sqrt(math.pow(x3-x1, 2) + math.pow(y3-y1, 2)) > 30:
			fingers.append(1)
	else:
		fingers.append(0)

	# 4 Fingers
	for id in range(1, 5):
		x1,x2 = getX(finger_landmarks[tipIds[id]].x) , getX(finger_landmarks[tipIds[id] - 2].x)
		y1,y2 = getX(finger_landmarks[tipIds[id]].y) , getX(finger_landmarks[tipIds[id] - 2].y)
		if y2 > y1 and math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2)) > 30:
			fingers.append(1)
		else:
			fingers.append(0)
	# print(fingers)
 
	# LIST OF VALUES FROM THUMB FINGER TO PINKY FINGER
	if fingers == [1, 1, 1, 1, 1]:
		return {"text": f"{NAME} says hello", "message": "Hello"}
	elif fingers == [1, 0, 0, 0, 0]:
		return {"text": f"{NAME} mentions the need for water", "message": "Need Water"}
	elif fingers == [0, 0, 0, 0, 1]:
		return {"text": f"{NAME} mentions the need to use the restroom", "message": "Washroom"}
	elif fingers == [0, 0, 1, 1, 1]:
		return {"text": f"{NAME} indicates feeling well", "message": "I'm Good"}
	elif fingers == [0, 0, 0, 0, 0]:
		return {"text": f"{NAME} mentions experiencing discomfort", "message": "In Pain"}
	# elif fingers == [1, 0, 0, 0, 1]:
	# 	return {"text": f"{NAME} indicates the need to make a call", "message": "Need to call"}
	return {}
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
	
def detect_emotion(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    # print(len(faces))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]  # Here label stores the emotion detected.
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            # print("no face")
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    # print("face detected")
    return frame
    
def detect_gesture(frame):
    global sleep_timer, draw, hands, curr_sign
    # frame = cv2.flip(frame, 1)	#flip horizontally(1) , as webcam images are mirrored
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = cv2.flip(frame, 1)
    op = hand_landmark.process(rgb)
    if curr_sign != None:
        cv2.putText(frame, curr_sign, (270,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)	#display selected name
    if op.multi_hand_landmarks == None:
        return frame
    i = op.multi_hand_landmarks[0]
    if not i.landmark:
        return frame
    
    draw.draw_landmarks(frame, i, hands.HAND_CONNECTIONS)
    
    if sleep_timer == MAX_SLEEP_TIME:
        curr_sign = None
        signText = getSign(i.landmark)
        if len(signText) != 0:
            print(signText["text"])
            curr_sign = signText["message"]
            cv2.putText(frame, curr_sign, (270,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)	#display selected name
            t = threading.Thread(target=speak, args=(signText["text"],))
            t.start()
            sleep_timer = 0
            
    return frame

while (True):
    # i+=1
    success, frame = cap.read()	#RETURN capture success/ failure , frame
    sleep_timer = min(MAX_SLEEP_TIME, sleep_timer + 1)
    if(not success):
        continue
    frame = cv2.flip(frame, 1)
    
    frame = detect_emotion(frame)
    frame = detect_gesture(frame)
    
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()