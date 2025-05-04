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
import gdown
from config import Config
from PIL import Image, ImageEnhance
from Utils.JWTtoken import token_required
from models.Frame import insert_frame_with_predictions
video_bp = Blueprint('video', __name__)

logging.basicConfig(level=logging.INFO)

class_labels = ['vertical', 'defect-free', 'hole', 'horizontal', 'lines', 'stain']
num_classes = len(class_labels)
CONFIDENCE_THRESHOLD = 0.6

GDRIVE_FILE_ID = Config.GDRIVE_MODEL_ID
if not GDRIVE_FILE_ID:
    raise EnvironmentError("❌ Environment variable 'GDRIVE_MODEL_ID' not set!")

MODEL_LOCAL_PATH = "./models/mixeddataset_resnet_classweights.pth"
os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)

def download_model_from_drive():
    if not os.path.exists(MODEL_LOCAL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        logging.info("⬇️ Downloading model from Google Drive...")
        gdown.download(url, MODEL_LOCAL_PATH, quiet=False)
        logging.info("✅ Model downloaded successfully.")

download_model_from_drive()

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
try:
    state_dict = torch.load(MODEL_LOCAL_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
    model = None

# ✨ Improved preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def enhance_edges(img):
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)
    return img

def preprocess_frame(frame):
    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)  # Denoise
        
        # Convert to PIL Image before enhancing
        frame_pil = Image.fromarray(frame)
        frame_pil = enhance_edges(frame_pil)
        
        return transform(frame_pil).unsqueeze(0)
    except Exception as e:
        logging.error(f"❌ Error during frame preprocessing: {e}")
        return None


def encode_image(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

@video_bp.route("/predict_video", methods=["POST"])
@token_required
def predict_video(current_user):
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

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    second = 0
    try:
        with torch.no_grad():
            while second < duration:
                cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
                success, frame = cap.read()
                if not success:
                    break

                # Resize frame if needed (model-specific size, e.g. 224x224)
                frame_resized = cv2.resize(frame, (224, 224))

                input_tensor = preprocess_frame(frame_resized)
                if input_tensor is None:
                    second += 1
                    continue

                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)

                if confidence.item() >= CONFIDENCE_THRESHOLD:
                    encoded = encode_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    confidence_score = round(confidence.item(), 3)
                    label = class_labels[predicted_class.item()]

                    results.append({
                        "frame": encoded,
                        "label": label,
                        "confidence": confidence_score,
                        "frame_number": int(second * fps)
                    })

                    frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
                    insert_frame_with_predictions(
                        user_id=current_user,
                        frame_data=frame_bytes,
                        keras_label=None,
                        keras_confidence=None,
                        pytorch_label=label,
                        pytorch_confidence=confidence_score
                    )

                second += 1
    finally:
        cap.release()
        os.remove(video_path)

    if not results:
        return jsonify({"message": "No significant defects detected!"}), 200

    return jsonify({"results": results})


@video_bp.route("/predict_frame", methods=["POST"])
@token_required
def predict_frame(current_user):
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "frame" not in request.files:
        return jsonify({"error": "No frame file provided"}), 400

    frame_file = request.files["frame"]
    print(f"Received frame file: {frame_file}")
    try:
        frame_bytes = frame_file.read()
        frame_np = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode frame"}), 400

        input_tensor = preprocess_frame(frame)
        print(f"Preprocessed frame shape: {input_tensor.shape}")
        if input_tensor is None:
            return jsonify({"error": "Failed to preprocess frame"}), 400

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

            if confidence.item() >= CONFIDENCE_THRESHOLD:
                result = {
                    "label": class_labels[predicted_class.item()],
                    "confidence": round(confidence.item(), 3),
                    "frame": encode_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                }
            else:
                result = {"message": "No significant defect detected in frame."}
        

        return jsonify(result)

    except Exception as e:
        logging.error(f"❌ Error processing frame: {e}")
        return jsonify({"error": str(e)}), 500
