from flask import Blueprint, render_template, redirect, url_for, send_file, request, current_app
from datetime import datetime
import os

bp = Blueprint('attendance', __name__)

@bp.route('/attendance')
def attendance():
    attendance_records = []
    today = datetime.now().strftime('%d-%B-%Y')
    attendance_path = current_app.config['ATTENDANCE_CSV']
    
    # Check if the file exists
    if not os.path.exists(attendance_path):
        # Create the file with a header
        with open(attendance_path, 'w') as f:
            f.write("Student Name,Date,Status\n")
    
    # Read the file content
    with open(attendance_path, 'r') as f:
        for line in f.readlines():
            attendance_records.append(line.strip().split(','))

    return render_template('attendance.html', 
                           attendance_records=attendance_records, 
                           class_name="Master Big Data & IoT", 
                           date=today)

@bp.route('/reset_attendance', methods=['POST'])
def reset_attendance():
    attendance_path = current_app.config['ATTENDANCE_CSV']
    with open(attendance_path, 'w') as f:
        f.write("Name, Time, Date")
    return redirect(url_for('attendance.attendance'))

@bp.route('/download_attendance', methods=['GET'])
def download_attendance():
    attendance_path = current_app.config['ATTENDANCE_CSV']
    today = datetime.now().strftime('%d-%B-%Y')
    return send_file(attendance_path, as_attachment=True, download_name=f'MBDIOT-{today}.csv')
