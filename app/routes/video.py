from flask import Blueprint, Response
from app.utils.encodings import generate_frames

bp = Blueprint('video', __name__)

@bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
