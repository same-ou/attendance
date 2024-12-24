import os
import pickle

ENCODINGS_FILE = r'C:\Users\DELL\Desktop\Attendance\face_encodings.pkl'
STUDENT_IMAGES_PATH = r'C:\Users\DELL\Desktop\Attendance\student images'

# # Function to check if student images have changed
# def images_changed():
#     last_mod_time = get_latest_image_mod_time()
#     if os.path.exists(ENCODINGS_FILE):
#         encodings_mod_time = os.path.getmtime(ENCODINGS_FILE)
#         print(f"{last_mod_time}  - {encodings_mod_time}")
#         return last_mod_time > encodings_mod_time
#     return True

# # Function to get the latest modification time of student images
# def get_latest_image_mod_time():
#     return max(os.path.getmtime(os.path.join(STUDENT_IMAGES_PATH, f)) for f in os.listdir(STUDENT_IMAGES_PATH))



# if images_changed():
#     print("yes")
# else:
#     print("hell naah")

print(os.path.exists(ENCODINGS_FILE))
print(f"Encodings file mod time: {os.path.getmtime(ENCODINGS_FILE) if os.path.exists(ENCODINGS_FILE) else 'Not Found'}")
print("Image files mod times:")
for f in os.listdir(STUDENT_IMAGES_PATH):
    print(f"{f}: {os.path.getmtime(os.path.join(STUDENT_IMAGES_PATH, f))}")
