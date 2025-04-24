from flask import Blueprint, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import io
import base64
import logging
from torchvision import models, transforms
import os
import tempfile

# Initialize Blueprint
video_bp = Blueprint('video', __name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the model for video processing
model = models.resnet18(pretrained=False)
num_classes = 6
model.fc = nn.Linear(model.fc.in_features, num_classes)
model_path = './models/mixeddataset_resnet_classweights.pth'
try:
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    logging.info("✅ Model 2 loaded successfully (PyTorch ResNet18).")
except Exception as e:
    logging.error(f"❌ Error loading Model 2: {e}")
    model2 = None

# Define class labels
class_labels = ['vertical', 'defect', 'hole', 'horizontal', 'lines', 'stain']

CONFIDENCE_THRESHOLD = 0.6

# Define image preprocessing for PyTorch
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_frame(frame):
    try:
        img_tensor = transform(frame).unsqueeze(0)  # Add batch dimension
        return img_tensor
    except Exception as e:
        logging.error(f"Error during frame preprocessing: {e}")
        return None

def encode_image(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

@video_bp.route("/predict_video", methods=["POST"])
def predict_video():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_video:
        video_path = temp_video.name
        video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Failed to read video file"}), 500

    frame_interval = 30  # Process every 30th frame
    results = []

    frame_count = 0
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

                confidence_val = confidence.item()
                predicted_label = class_labels[predicted_class.item()]

                if confidence_val >= CONFIDENCE_THRESHOLD:
                    encoded_frame = encode_image(frame)
                    results.append({
                        "frame": encoded_frame,
                        "label": predicted_label,
                        "confidence": round(confidence_val, 3)
                    })

        frame_count += 1

    cap.release()
    os.remove(video_path)

    return jsonify({"results": results})
