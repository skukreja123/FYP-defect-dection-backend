import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # General configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'mlwmalmlwmlwmalmlwmlwmlwmlwmlw')
    DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
    DB_URI = os.getenv('DATABASE_URL', 'sqlite:///default.db')