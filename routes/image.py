from flask import Blueprint, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io
import base64
import logging
from PIL import Image
from torchvision import models, transforms
import json
import urllib.request
import os
from dotenv import load_dotenv
from config import Config

# Load environment variables
load_dotenv()

# Initialize Blueprint
image_bp = Blueprint('image', __name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# ---------------- Model Download from AWS S3 ----------------
MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

pth_url = Config.S3_PTH_MODEL_URL

pth_model_path = os.path.join(MODEL_DIR, 'resnet_model.pth')

if not pth_url:
    raise EnvironmentError("❌ S3 PTH model URL not set in .env")

def download_file(url, local_path):
    if not os.path.exists(local_path):
        logging.info(f"⬇️ Downloading {local_path} from S3...")
        urllib.request.urlretrieve(url, local_path)
        logging.info(f"✅ Downloaded {local_path} successfully.")

download_file(pth_url, pth_model_path)

# ---------------- Load PyTorch Model ----------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 6)

try:
    state_dict = torch.load(pth_model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    logging.info("✅ PyTorch model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading PyTorch model: {e}")
    model = None

# ---------------- ImageNet Pretrained Model for Cloth Detection ----------------
imagenet_model = models.resnet18(pretrained=True)
imagenet_model.eval()

LABELS_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with urllib.request.urlopen(LABELS_URL) as url:
    class_idx = json.load(url)
    imagenet_labels = [class_idx[str(k)][1] for k in range(len(class_idx))]

imagenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------- Class Labels ----------------
class_labels = ['vertical', 'defect', 'hole', 'horizontal', 'lines', 'stain']
CONFIDENCE_THRESHOLD = 0.6

# ---------------- Cloth Detection ----------------
def is_cloth_by_imagenet(img: Image.Image, allowed_keywords=[
            'suit', 'shirt', 'jean', 'tshirt', 'fabric', 'apparel',
            'sock', 'pajama', 'trouser', 'shorts', 'cloth', 'jacket',
            'sweater', 'dress', 'skirt', 'kurta', 'blazer', 'undergarment',
            'hoodie', 'vest', 'tracksuit', 'uniform', 'tick'
        ]):
    transformed = imagenet_transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = imagenet_model(transformed)
        probs = F.softmax(outputs[0], dim=0)
        top5 = torch.topk(probs, 5)
        top_classes = [imagenet_labels[i] for i in top5.indices]

        logging.info(f"🧵 Top-5 ImageNet labels: {top_classes}")

        return any(keyword in label.lower() for label in top_classes for keyword in allowed_keywords)

# ---------------- Image Preprocessing ----------------
def preprocess_image(image_data):
    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1]))).convert('RGB')
        img_resized = img.resize((64, 64))
        img_array = np.array(img_resized) / 255.0
        return img_array, img
    except Exception as e:
        logging.error(f"❌ Error during image preprocessing: {e}")
        raise ValueError("Error processing the image.")

# ---------------- Prediction Route ----------------
@image_bp.route('/predict_image', methods=['POST'])
def predict_image():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400

        image_data = data['image']
        img_array, img_pil = preprocess_image(image_data)

        if not is_cloth_by_imagenet(img_pil):
            return jsonify({'error': 'Invalid image: no cloth detected'}), 400

        # --- PyTorch Prediction ---
        torch_input = torch.tensor(img_array.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction = model(torch_input)
        confidence, idx = torch.max(prediction, 1)
        label = class_labels[idx.item()]
        confidence = float(confidence.item())

        result = {
            'model': {'label': label, 'confidence': confidence} if confidence >= CONFIDENCE_THRESHOLD else None
        }

        return jsonify(result)

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logging.error(f"❌ Internal error: {e}")
        return jsonify({'error': 'Internal server error'}), 500
