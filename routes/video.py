from flask import Blueprint, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import io
import base64
import logging
import os
import tempfile
from config import Config

# Initialize Blueprint
video_bp = Blueprint('video', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
class_labels = ['vertical', 'defect-free', 'hole', 'horizontal', 'lines', 'stain']
num_classes = len(class_labels)
CONFIDENCE_THRESHOLD = 0.6

# Local model path
MODEL_LOCAL_PATH = "./models/mixeddataset_resnet_classweights.pth"
os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)

# Load model directly from local path
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

try:
    state_dict = torch.load(MODEL_LOCAL_PATH, map_location='cpu')  # Move to GPU if available
    model.load_state_dict(state_dict)
    model.eval()
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
    model = None

# Frame preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

def preprocess_frame(frame):
    try:
        return transform(frame).unsqueeze(0)
    except Exception as e:
        logging.error(f"❌ Error during frame preprocessing: {e}")
        return None

def encode_image(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

# Route for video prediction
@video_bp.route("/predict_video", methods=["POST"])
def predict_video():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        video_path = temp_video.name
        video_file.save(video_path)

    results = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        os.remove(video_path)
        return jsonify({"error": "Failed to read video file"}), 500

    frame_interval = 15  # Adjusting frame interval to process every 15th frame
    frame_count = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % frame_interval == 0:
                input_tensor = preprocess_frame(frame)
                if input_tensor is None:
                    continue

                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)

                    if confidence.item() >= CONFIDENCE_THRESHOLD:
                        results.append({
                            "frame": encode_image(frame),
                            "label": class_labels[predicted_class.item()],
                            "confidence": round(confidence.item(), 3)
                        })

            frame_count += 1
    finally:
        cap.release()
        os.remove(video_path)

    if not results:
        return jsonify({"message": "No confident predictions found"}), 200

    return jsonify({"results": results})

