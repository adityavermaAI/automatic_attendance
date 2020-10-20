import numpy as np
from cv2 import cv2
import os
import face_recognition
from datetime import datetime

def getEncodings(met_list):
    encodings = []
    for met in met_list:
        met_rgb = cv2.cvtColor(met, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(met_rgb)[0]
        encodings.append(encode)
    return(encodings)

def mark_attendance(name):
    name_list = []
    with open('attendance.csv','r+') as f:
        lines = f.readlines()

        for line in lines:
            entry = line.split(',')
            name_list = entry[0]
        
        if name not in name_list:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')



path = 'faces'
people_met = []
people_names = []
people_names_with_ex = os.listdir(path)

for person in people_names_with_ex:
    person_m = cv2.imread(f'{path}/{person}')
    people_met.append(person_m)
    people_names.append(os.path.splitext(person)[0])

print(people_names)
print('NO. of known faces =', len(people_names))
people_encodings = getEncodings(people_met)
print('ENCODING COMPLETE')



cap = cv2.VideoCapture(0)
while True:
    flag, cam_image = cap.read()
    faces_loc_cur_frame = face_recognition.face_locations(cam_image)
    encodings_cur_frame = face_recognition.face_encodings(cam_image, faces_loc_cur_frame)

    for en_cur_frame, loc_cur_frame in zip(encodings_cur_frame, faces_loc_cur_frame):
        matches = face_recognition.compare_faces(people_encodings, en_cur_frame, 0.5)
        face_dis = face_recognition.face_distance(people_encodings, en_cur_frame)
        match_index = np.argmin(face_dis)
        if matches[match_index]:
            name = people_names[match_index]

            y1, x2, y2, x1 = loc_cur_frame
            cv2.rectangle(cam_image, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(cam_image, (x1,y2+35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(cam_image, name, (x1+6,y2+25), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 2)
            mark_attendance(name)


    cv2.imshow('camera', cam_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()