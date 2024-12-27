import cv2
import face_recognition
import os
import random
import pickle
import numpy as np
from matplotlib import pyplot as plt

# Configuration
STUDENT_IMAGES_PATH = 'app/data/test images'
ENCODINGS_FILE = 'app/data/face_encodings.pkl'
NAMES_FILE = 'app/data/face_encodings_names.pkl'
CONFIDENCE_THRESHOLD = 0.6
NUM_TEST_IMAGES = 21  

# Load encodings and class names
def load_encodings():
    with open(ENCODINGS_FILE, 'rb') as f:
        encoded_face_train = pickle.load(f)
    with open(NAMES_FILE, 'rb') as f:
        classNames = pickle.load(f)
    return encoded_face_train, classNames

# Preprocess images with random transformations (including blur and lighting changes)
def preprocess_image(img):
    transformations = [
        # Brightness/Contrast changes to simulate lighting variations (moderate)
        lambda x: cv2.convertScaleAbs(x, alpha=random.uniform(0.8, 1.3), beta=random.randint(-40, 40)),  # Moderate Brightness/Contrast
        
        # Apply Gaussian blur (simulating slight focus issues or low-quality lens)
        lambda x: cv2.GaussianBlur(x, (5, 5), random.uniform(0, 2)),  # Light blur to simulate mild focus issues
        
        # Motion blur (simulating camera shake or moving subject)
        lambda x: cv2.blur(x, (random.randint(3, 7), random.randint(3, 7))),  # Apply slight motion blur
        
        # Brightness shift (mimicking overexposure or underexposure)
        lambda x: cv2.convertScaleAbs(x, alpha=1, beta=random.randint(-60, 60)),  # Brightness shift
        
        # Random pixelation effect (simulating compression artifacts or low resolution)
        lambda x: cv2.resize(cv2.resize(x, (random.randint(50, 100), random.randint(50, 100))), 
                             (x.shape[1], x.shape[0]), interpolation=cv2.INTER_LINEAR)
    ]
    
    # Shuffle transformations and apply a random subset
    random.shuffle(transformations)
    for transform in transformations[:random.randint(1, len(transformations))]:
        img = transform(img)
        
    return img

# Select random images from the folder
def get_random_images(folder_path, num_images):
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    return random.sample(images, min(num_images, len(images)))

# Display image with results (Bounding box and consistent font size)
def display_image_with_results(img, name, face_distance, is_recognized, faceloc, closest_match=None):
    face_distance_text = f"{face_distance:.4f}" if isinstance(face_distance, (float, int)) else "-"
    is_recognized_text = str(is_recognized)

    # Create bounding box for the face
    y1, x2, y2, x1 = faceloc
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0) if is_recognized else (0, 0, 255), 2)

    # Background for text
    padding = 10
    box_height = 100
    cv2.rectangle(img, (x1, y2), (x2, y2 + box_height), (0, 255, 0) if is_recognized else (0, 0, 255), cv2.FILLED)

    # Text to display with consistent font size
    result_text = (
        f"Name: {name}\n"
        f"Face Distance: {face_distance_text}\n"
        f"Threshold: {CONFIDENCE_THRESHOLD}\n"
        f"Recognized: {is_recognized_text}\n"
    )
    
    if closest_match and not is_recognized:  # Only show closest match if not recognized
        result_text += f"Closest Match: {closest_match[0]} (Face Distance: {closest_match[1]:.4f})"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7  # Fixed font size
    y = y2 + padding
    for line in result_text.split('\n'):
        cv2.putText(img, line, (x1 + padding, y), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        y += 30
    print(result_text)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Test face recognition
def test_face_recognition(encoded_face_train, classNames):
    random_images = get_random_images(STUDENT_IMAGES_PATH, NUM_TEST_IMAGES)

    # Initialize counters
    recognized_count = 0
    unknown_count = 0
    not_detected_count = 0

    for img_path in random_images:
        print(f"Processing image: {img_path}")
        img = cv2.imread(img_path)
        original_img = img.copy()

        # Preprocess the image
        img = preprocess_image(img)

        # Detect faces in the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces_in_frame = face_recognition.face_locations(img_rgb)
        encoded_faces = face_recognition.face_encodings(img_rgb, faces_in_frame)

        if not faces_in_frame:
            print("No face detected in this image.")
            not_detected_count += 1
            display_image_with_results(img, "No Face Detected", "-", False, (0, 0, 0, 0))
            continue

        for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
            matchIndex = np.argmin(faceDist)
            is_recognized = faceDist[matchIndex] < CONFIDENCE_THRESHOLD

            if is_recognized:
                recognized_count += 1
            else:
                unknown_count += 1

            name = classNames[matchIndex].upper() if is_recognized else "Unknown"
            
            # Find the closest match (minimum face distance)
            closest_match_index = np.argmin(faceDist)
            closest_name = classNames[closest_match_index].upper()
            closest_distance = faceDist[closest_match_index]

            # Display bounding box with text and closest match if not recognized
            display_image_with_results(original_img, name, faceDist[matchIndex], is_recognized, faceloc, 
                                       closest_match=(closest_name, closest_distance) if not is_recognized else None)

    # Display final counts
    print(f"\nRecognition Summary:")
    print(f"Recognized faces: {recognized_count}")
    print(f"Unknown faces: {unknown_count}")
    print(f"Faces not detected: {not_detected_count}")

# Main execution
if __name__ == "__main__":
    # Load encodings
    encoded_face_train, classNames = load_encodings()

    # Test recognition
    test_face_recognition(encoded_face_train, classNames)
