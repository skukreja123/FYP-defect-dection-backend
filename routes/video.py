# BLUEPRINT: video_bp (video prediction)
from flask import Blueprint, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import io, os, tempfile, base64, logging
from config import Config

video_bp = Blueprint('video', __name__)
logging.basicConfig(level=logging.INFO)

# Settings
MODEL_PATH = "./models/mixeddataset_resnet_classweights.pth"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
CLASS_LABELS = ['vertical', 'defect', 'hole', 'horizontal', 'lines', 'stain']
CONFIDENCE_THRESHOLD = 0.6
FRAME_INTERVAL = 30

# Download model
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={Config.GDRIVE_MODEL_ID}", MODEL_PATH, quiet=False)

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_LABELS))
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    logging.info("✅ Video model loaded.")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    model = None

# Frame transform
frame_transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((64, 64)),
    transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
])

def preprocess_frame(frame):
    try:
        return frame_transform(frame).unsqueeze(0)
    except Exception as e:
        logging.error(f"❌ Frame preprocess failed: {e}")
        return None

def encode_image(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

@video_bp.route('/predict_video', methods=['POST'])
def predict_video():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files['video']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name
        video_file.save(video_path)

    results = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        os.remove(video_path)
        return jsonify({"error": "Could not open video"}), 500

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % FRAME_INTERVAL == 0:
                input_tensor = preprocess_frame(frame)
                if input_tensor is None:
                    continue
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    conf, idx = torch.max(probs, 1)
                    if conf.item() >= CONFIDENCE_THRESHOLD:
                        results.append({
                            "frame": encode_image(frame),
                            "label": CLASS_LABELS[idx.item()],
                            "confidence": round(conf.item(), 3)
                        })
            frame_count += 1
    finally:
        cap.release()
        os.remove(video_path)

    if not results:
        return jsonify({"message": "No confident predictions found"}), 200

    return jsonify({"results": results})

