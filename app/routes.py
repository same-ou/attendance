from flask import Blueprint, render_template, Response, redirect, url_for, send_file
from .utils import generate_frames, reset_attendance, download_attendance, markAttendance
from datetime import datetime
import os

bp = Blueprint('routes', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_running
    camera_running = True
    return redirect(url_for('routes.index'))

@bp.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_running
    camera_running = False
    return redirect(url_for('routes.index'))

@bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/attendance')
def attendance():
    today = datetime.now().strftime('%d-%B-%Y')
    attendance_records = []
    attendance_path = os.getenv('ATTENDANCE_FILE')
    with open(attendance_path, 'r') as f:
        for line in f.readlines():
            attendance_records.append(line.strip().split(','))
    return render_template('attendance.html', attendance_records=attendance_records, date=today)

@bp.route('/reset_attendance', methods=['POST'])
def reset_attendance_route():
    reset_attendance()
    return redirect(url_for('routes.attendance'))

@bp.route('/download_attendance', methods=['GET'])
def download_attendance_route():
    return download_attendance()
