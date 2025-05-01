import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # General configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'mlwmalmlwmlwmalmlwmlwmlwmlwmlw')
    DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
    DB_URI = os.getenv('DATABASE_URL', 'sqlite:///default.db')
    GDRIVE_MODEL_ID = os.getenv('GDRIVE_MODEL_ID', 'default_model_id')
    h5py_model_ID = os.getenv('H5PY_MODEL_ID', 'default_h5py_model_id')
    PORT = int(os.getenv('PORT', 5000))
    S3_PTH_MODEL_URL = os.getenv('S3_PTH_MODEL_URL', 'default_s3_pth_model_url')
    S3_H5_MODEL_URL = os.getenv('S3_H5_MODEL_URL', 'default_s3_h5_model_url')