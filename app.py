from flask import Flask, render_template, Response, redirect, url_for
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle

app = Flask(__name__)

# Path to student images
path = './student images'
encodings_file = './face_encodings.pkl'
attendance_path = './Attendance.csv'

# Encoding the images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList

# Load or Create Encodings
if os.path.exists(encodings_file):
    print("Loading saved encodings...")
    with open(encodings_file, 'rb') as f:
        encoded_face_train = pickle.load(f)
        with open(encodings_file.replace(".pkl", "_names.pkl"), 'rb') as name_file:
            classNames = pickle.load(name_file)
else:
    print("Encoding images...")
    images = []
    classNames = []
    mylist = os.listdir(path)
    for cl in mylist:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    encoded_face_train = findEncodings(images)
    # Save encodings to file
    with open(encodings_file, 'wb') as f:
        pickle.dump(encoded_face_train, f)
    with open(encodings_file.replace(".pkl", "_names.pkl"), 'wb') as f:
        pickle.dump(classNames, f)
    print("Encodings saved to file.")

# Attendance function
def markAttendance(name):
    with open(attendance_path, 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'\n{name}, {time}, {date}')

# Video Stream Generator
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faces_in_frame = face_recognition.face_locations(imgS)
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

        for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
            matches = face_recognition.compare_faces(encoded_face_train, encode_face)
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
            matchIndex = np.argmin(faceDist)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        # Convert to byte stream for Flask
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def attendance():
    attendance_records = []
    with open(attendance_path, 'r') as f:
        for line in f.readlines():
            attendance_records.append(line.strip().split(','))
    return render_template('attendance.html', attendance_records=attendance_records)

if __name__ == '__main__':
    app.run(debug=True)
