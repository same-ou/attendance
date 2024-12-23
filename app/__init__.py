from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config.from_mapping(
        ATTENDANCE_CSV='../instance/Attendance.csv',
        ENCODINGS_FILE='../face_encodings.pkl',
        STUDENT_IMAGES='../student images',
    )

    from app.routes import attendance, video, camera_control, index
    app.register_blueprint(attendance.bp)
    app.register_blueprint(video.bp)
    app.register_blueprint(camera_control.bp)
    app.register_blueprint(index.bp)

    return app
