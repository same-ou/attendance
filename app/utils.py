import os
import cv2
import pickle
import face_recognition
import numpy as np
from datetime import datetime
from flask import send_file

camera_running = True

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList

def load_encodings(encodings_file, student_images_path):
    """
    Loads the face encodings and names from files or generates them if the files don't exist.

    Args:
        encodings_file (str): Path to the file storing face encodings.
        student_images_path (str): Path to the directory containing student images.

    Returns:
        tuple: A tuple containing the list of encodings and their corresponding class names.
    """
    # Check if encoding files exist
    if os.path.exists(encodings_file) and os.path.exists(encodings_file.replace(".pkl", "_names.pkl")):
        print("Loading saved encodings...")
        with open(encodings_file, 'rb') as f:
            encoded_face_train = pickle.load(f)
        with open(encodings_file.replace(".pkl", "_names.pkl"), 'rb') as name_file:
            class_names = pickle.load(name_file)
    else:
        print("Encoding images...")
        images = []
        class_names = []
        student_list = os.listdir(student_images_path)
        for student in student_list:
            student_img = cv2.imread(f'{student_images_path}/{student}')
            images.append(student_img)
            class_names.append(os.path.splitext(student)[0])

        # Generate encodings
        encoded_face_train = findEncodings(images)

        # Save encodings and class names to files
        with open(encodings_file, 'wb') as f:
            pickle.dump(encoded_face_train, f)
        with open(encodings_file.replace(".pkl", "_names.pkl"), 'wb') as name_file:
            pickle.dump(class_names, name_file)
        print("Encodings saved.")

    return encoded_face_train, class_names

def markAttendance(name):
    attendance_path = os.getenv('ATTENDANCE_FILE')
    with open(attendance_path, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'\n{name}, {time}, {date}')

def reset_attendance():
    attendance_path = os.getenv('ATTENDANCE_FILE')
    with open(attendance_path, 'w') as f:
        f.write("Name, Time, Date")

def download_attendance():
    attendance_path = os.getenv('ATTENDANCE_FILE')
    today = datetime.now().strftime('%d-%B-%Y')
    return send_file(attendance_path, as_attachment=True, download_name=f'Attendance-{today}.csv')

def generate_frames():
    global camera_running
    encodings_file = os.getenv('ENCODINGS_FILE')
    student_images_path = os.getenv('STUDENT_IMAGES_PATH')

    # Load encodings and names
    encoded_face_train, classNames = load_encodings(encodings_file, student_images_path)

    cap = cv2.VideoCapture(0)
    while camera_running:
        success, img = cap.read()
        if not success:
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faces_in_frame = face_recognition.face_locations(imgS)
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

        for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
            matches = face_recognition.compare_faces(encoded_face_train, encode_face, tolerance=0.5)
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
            matchIndex = np.argmin(faceDist)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = [v * 4 for v in faceloc]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


