# import cv2
# import face_recognition
# import os
# import pickle
# import numpy as np
# from app.utils.attendance import mark_attendance  # Assuming this function is defined in attendance.py

# # Paths to resources
# STUDENT_IMAGES_PATH = '../student images'
# ENCODINGS_FILE = '../face_encodings.pkl'
# NAMES_FILE = '../face_encodings_names.pkl'

# # Global variables for storing encodings and class names
# encoded_face_train, classNames = None, None
# camera_running = True  # State for Play/Stop camera

# # Function to check if student images have changed
# def images_changed():
#     last_mod_time = get_latest_image_mod_time()
#     if os.path.exists(ENCODINGS_FILE):
#         encodings_mod_time = os.path.getmtime(ENCODINGS_FILE)
#         return last_mod_time > encodings_mod_time
#     return True

# # Function to get the latest modification time of student images
# def get_latest_image_mod_time():
#     return max(os.path.getmtime(os.path.join(STUDENT_IMAGES_PATH, f)) for f in os.listdir(STUDENT_IMAGES_PATH))

# # Function to load existing encodings from pickle files
# def load_existing_encodings():
#     global encoded_face_train, classNames
#     with open(ENCODINGS_FILE, 'rb') as f:
#         encoded_face_train = pickle.load(f)
#     with open(NAMES_FILE, 'rb') as name_file:
#         classNames = pickle.load(name_file)
#     print("Existing encodings loaded.")

# # Function to save new encodings to pickle files
# def save_encodings(images, class_names, encoded_face_train):
#     with open(ENCODINGS_FILE, 'wb') as f:
#         pickle.dump(encoded_face_train, f)
#     with open(NAMES_FILE, 'wb') as f:
#         pickle.dump(class_names, f)
#     print("New encodings saved.")

# # Function to find encodings of images
# def find_encodings(images):
#     encode_list = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encoded_face = face_recognition.face_encodings(img)[0]
#         encode_list.append(encoded_face)
#     return encode_list

# # Function to load or create encodings based on image folder changes
# def load_encodings():
#     global encoded_face_train, classNames

#     if images_changed():
#         print("Images have changed, generating new encodings...")
#         images, class_names = load_student_images()
#         encoded_face_train = find_encodings(images)
#         save_encodings(images, class_names, encoded_face_train)
#     else:
#         load_existing_encodings()

#     return encoded_face_train, classNames

# # Function to load student images and class names from the images folder
# def load_student_images():
#     images = []
#     class_names = []
#     mylist = os.listdir(STUDENT_IMAGES_PATH)
#     for cl in mylist:
#         cur_img = cv2.imread(f'{STUDENT_IMAGES_PATH}/{cl}')
#         images.append(cur_img)
#         class_names.append(os.path.splitext(cl)[0])
#     return images, class_names

# # Function to handle face detection and recognition in the video stream
# def detect_faces_in_frame(img, encoded_face_train, classNames):
#     imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#     faces_in_frame = face_recognition.face_locations(imgS)
#     encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
#     return faces_in_frame, encoded_faces

# # Function to process face recognition matches and draw boxes on faces
# def process_face_matches(encoded_faces, faces_in_frame, encoded_face_train, classNames, img):
#     for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
#         matches = face_recognition.compare_faces(encoded_face_train, encode_face)
#         faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
#         matchIndex = np.argmin(faceDist)

#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper()
#             draw_face_box(faceloc, img, name)
#             mark_attendance(name)  # Mark attendance for recognized face

# # Function to draw a box and label around the detected face
# def draw_face_box(faceloc, img, name):
#     y1, x2, y2, x1 = faceloc
#     y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
#     cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

# # Video Stream Generator
# def generate_frames():
#     global camera_running, encoded_face_train, classNames

#     # Load known faces at the beginning of the stream (This will check for changes and update encodings if needed)
#     encoded_face_train, classNames = load_encodings()

#     cap = cv2.VideoCapture(0)

#     while camera_running:
#         success, img = cap.read()
#         if not success:
#             break

#         # Detect and process faces
#         faces_in_frame, encoded_faces = detect_faces_in_frame(img, encoded_face_train, classNames)
#         process_face_matches(encoded_faces, faces_in_frame, encoded_face_train, classNames, img)

#         # Encode frame to send to browser
#         _, buffer = cv2.imencode('.jpg', img)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
#     cap.release()



import cv2
from flask import Response

def generate_frames():
    # Open the camera
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        # Stream the frame to the browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
