import sys
import time
import cv2
import face
import urllib
import urllib.request
import numpy as np
import os
import pickle
import pandas as pd


classifier_filename="./my_classifier.pkl"
classifier_filename_exp = os.path.expanduser(classifier_filename)
with open(classifier_filename_exp, 'rb') as infile:
    (model, class_names) = pickle.load(infile)

attend=[]
attendance=[]
class_names.append('Unknown')

def add_overlays(frame, faces, frame_rate):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,(face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),(0, 255, 0), 2)
            cv2.rectangle(frame, (face_bb[0], face_bb[3] - 35), (face_bb[2], face_bb[3]), (0, 0, 255), cv2.FILLED)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0]+6, face_bb[3]-6),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 255, 225),1)
                if face.name not in attend:
                    attend.append(face.name)

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)

def start_recognition(mode='webcam'):
    if mode=='webcam':
        frame_interval = 3  # Number of frames after which to run face detection
        fps_display_interval = 5  # seconds
        frame_rate = 0
        frame_count = 0

        video_capture = cv2.VideoCapture(0)
        face_recognition = face.Recognition()
        start_time = time.time()

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            if (frame_count % frame_interval) == 0:
                faces = face_recognition.identify(frame)

                # Check our current fps
                end_time = time.time()
                if (end_time - start_time) > fps_display_interval:
                    frame_rate = int(frame_count / (end_time - start_time))
                    start_time = time.time()
                    frame_count = 0

            add_overlays(frame, faces, frame_rate)

            frame_count += 1
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()
    else:
        a= input('Enter IP Address = ')
        url= 'http://'+str(a)+'/shot.jpg'
        frame_interval = 3  # Number of frames after which to run face detection
        fps_display_interval = 5  # seconds
        frame_rate = 0
        frame_count = 0
        face_recognition = face.Recognition()
        start_time = time.time()
        while True:
            try:
                imgResp = urllib.request.urlopen(url, timeout=10) 
                imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
            except:
                print('socket timed out - URL %s , Please Make Sure you are sharing same IP or Hotspot within devices', url)
                break
            #with urllib.request.urlopen(url, timeout=10) as imgResp:
            #    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
            frame=cv2.imdecode(imgNp,-1)
            if (frame_count % frame_interval) == 0:
                faces = face_recognition.identify(frame)

                # Check our current fps
                end_time = time.time()
                if (end_time - start_time) > fps_display_interval:
                    frame_rate = int(frame_count / (end_time - start_time))
                    start_time = time.time()
                    frame_count = 0

            add_overlays(frame, faces, frame_rate)

            frame_count += 1
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        cv2.destroyAllWindows()
        #print('attend)
        for i in class_names:
            if i in attend:
                attendance.append('PRESENT')
            else:
                attendance.append('ABSENT')
        df = pd.DataFrame()
        df['Student Name']= class_names
        df['Attendance']= attendance
        print('present students = ',attend)
        print("###############################################")
        print(df)
        print("###############################################")
        print("Writing the attendance list to CSV file")
        df.to_csv('attendance.csv')
        print('done!!!!')
        