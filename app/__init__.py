from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

def create_app():
    app = Flask(__name__)

    # Import and register blueprints
    from .routes import bp as routes_bp
    app.register_blueprint(routes_bp)

    return app