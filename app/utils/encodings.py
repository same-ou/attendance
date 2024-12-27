import cv2
import face_recognition
import os
import pickle
import numpy as np
from app.utils.attendance import mark_attendance  # Assuming this function is defined in attendance.py
import app.routes.camera_control as camera_control
from collections import defaultdict

# Paths to resources
STUDENT_IMAGES_PATH = 'app/data/student images'
ENCODINGS_FILE = 'app/data/face_encodings.pkl'
NAMES_FILE = 'app/data/face_encodings_names.pkl'
ATTENDANCE_PATH = 'instance/Attendance.csv'

# Global variables for storing encodings and class names
encoded_face_train, classNames = None, None

# Track persistent recognition (across multiple frames)
face_recognition_history = defaultdict(int)
recognition_threshold = 20  # Number of consecutive frames needed for a valid recognition

# Function to check if student images have changed
def images_changed():
    last_mod_time = get_latest_image_mod_time()
    if os.path.exists(ENCODINGS_FILE):
        encodings_mod_time = os.path.getmtime(ENCODINGS_FILE)
        return last_mod_time > encodings_mod_time
    return True

# Function to get the latest modification time of student images
def get_latest_image_mod_time():
    return max(os.path.getmtime(os.path.join(STUDENT_IMAGES_PATH, f)) for f in os.listdir(STUDENT_IMAGES_PATH))

# Function to load existing encodings from pickle files
def load_existing_encodings():
    global encoded_face_train, classNames
    with open(ENCODINGS_FILE, 'rb') as f:
        encoded_face_train = pickle.load(f)
    with open(NAMES_FILE, 'rb') as name_file:
        classNames = pickle.load(name_file)
    print("Existing encodings loaded.")

# Function to save new encodings to pickle files
def save_encodings(images, class_names, encoded_face_train):
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(encoded_face_train, f)
    with open(NAMES_FILE, 'wb') as f:
        pickle.dump(class_names, f)
    print("New encodings saved.")

# Function to find encodings of images
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encode_list.append(encoded_face)
    return encode_list

# Function to load or create encodings based on image folder changes
def load_encodings():
    global encoded_face_train, classNames

    if images_changed():
        print("Images have changed, generating new encodings...")
        images, class_names = load_student_images()
        encoded_face_train = find_encodings(images)
        save_encodings(images, class_names, encoded_face_train)
    else:
        load_existing_encodings()

    return encoded_face_train, classNames

# Function to load student images and class names from the images folder
def load_student_images():
    images = []
    class_names = []
    mylist = os.listdir(STUDENT_IMAGES_PATH)
    for cl in mylist:
        cur_img = cv2.imread(f'{STUDENT_IMAGES_PATH}/{cl}')
        images.append(cur_img)
        class_names.append(os.path.splitext(cl)[0])
    return images, class_names

# Function to handle face detection and recognition in the video stream
def detect_faces_in_frame(img, encoded_face_train, classNames):
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    return faces_in_frame, encoded_faces

# Function to draw a box and label around the detected face
def draw_face_box(faceloc, img, name):
    y1, x2, y2, x1 = faceloc
    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

# Function to display advice on the frame (when face is detected)
def display_advice(img, message):
    cv2.putText(img, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Function to process and track face recognition
def process_face_matches(encoded_faces, faces_in_frame, encoded_face_train, classNames, img):
    current_recognized_faces = []
    
    if not faces_in_frame:
        display_advice(img, "No face detected. Please stand in front of the camera.")
        return

    # Process each face and recognize it
    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        is_recognized = faceDist[matchIndex] < 0.6  # Confidence threshold for recognition

        if is_recognized:
            name = classNames[matchIndex].upper()
            current_recognized_faces.append((name, faceloc))
            
            # Track persistent recognition
            face_recognition_history[name] += 1
            if face_recognition_history[name] >= recognition_threshold:
                mark_attendance(name, ATTENDANCE_PATH)
                show_alert(f"{name} is registered.")
                face_recognition_history.clear()  # Reset after marking attendance
        else:
            name = "Unknown"
            display_advice(img, "Please center your face and stand straight.")
            current_recognized_faces.append((name, faceloc))

    # Draw bounding box and label for recognized faces
    for recognized_face in current_recognized_faces:
        name, faceloc = recognized_face
        draw_face_box(faceloc, img, name)

# Function to display alert
def show_alert(message):
    print(message)  # You can replace this with a real alert system, e.g., a UI popup

# Global variable to track if encodings are already loaded
encodings_loaded = False

def generate_frames():
    global encoded_face_train, classNames, encodings_loaded

    # Load known faces only the first time
    if not encodings_loaded:
        print("Loading encodings...")
        encoded_face_train, classNames = load_encodings()
        encodings_loaded = True
        print("Encodings loaded successfully.")

    cap = cv2.VideoCapture(0)

    while True:
        if not camera_control.camera_running:  # Stop generating frames if the camera is not running
            cap.release()
            break

        success, img = cap.read()
        if not success:
            break

        # Detect faces and process recognition
        faces_in_frame, encoded_faces = detect_faces_in_frame(img, encoded_face_train, classNames)
        process_face_matches(encoded_faces, faces_in_frame, encoded_face_train, classNames, img)

        # Encode frame to send to browser
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
