from flask import Flask

import os

def create_app():
    app = Flask(__name__)

    # Get the absolute path to the project root
    base_dir = os.path.abspath(os.path.dirname(__file__))  # Get the path of the current directory (app/)
    
    # Set absolute paths based on the project root
    app.config.from_mapping(
        ATTENDANCE_CSV=os.path.join(base_dir, '..', 'instance', 'Attendance.csv'),
        ENCODINGS_FILE=os.path.join(base_dir, '..', 'app', 'data', 'face_encodings.pkl'),
        STUDENT_IMAGES=os.path.join(base_dir, '..', 'app', 'data', 'student images'),
    )

    # Import and register blueprints
    from app.routes import attendance, video, camera_control, index
    app.register_blueprint(attendance.bp)
    app.register_blueprint(video.bp)
    app.register_blueprint(camera_control.bp)
    app.register_blueprint(index.bp)

    return app

