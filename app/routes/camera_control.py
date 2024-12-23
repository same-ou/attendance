from flask import Blueprint, redirect, url_for

bp = Blueprint('camera_control', __name__)

camera_running = True

@bp.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_running
    camera_running = True
    return redirect(url_for('index.home')) 

@bp.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_running
    camera_running = False
    return redirect(url_for('index.home'))  
