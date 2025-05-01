# BLUEPRINT: image_bp (image prediction)
from flask import Blueprint, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np

import io
import base64
import logging
from PIL import Image, ImageEnhance
from torchvision import models, transforms
import json
import urllib.request
import os

import gdown
from dotenv import load_dotenv
import urllib.request

# Load .env variables
load_dotenv()

# Flask Blueprint
image_bp = Blueprint('image', __name__)

# Logging
logging.basicConfig(level=logging.INFO)

# Paths and constants
MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)
pth_path = os.path.join(MODEL_DIR, 'resnet_model.pth')
h5_path = os.path.join(MODEL_DIR, 'keras_model.h5')
CONFIDENCE_THRESHOLD = 0.6

# Download models
if not os.path.exists(pth_path):
    gdown.download(f"https://drive.google.com/uc?id={Config.GDRIVE_MODEL_ID}", pth_path, quiet=False)

if not os.path.exists(h5_path):
    gdown.download(f"https://drive.google.com/uc?id={Config.h5py_model_ID}", h5_path, quiet=False)

# Load models
try:
    model1 = tf.keras.models.load_model(h5_path, compile=False)
    logging.info("✅ Keras model loaded.")
except Exception as e:
    logging.error(f"❌ Keras model load error: {e}")
    model1 = None

model2 = models.resnet18(pretrained=False)
model2.fc = nn.Linear(model2.fc.in_features, 6)
try:
    model2.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model2.eval()
    logging.info("✅ PyTorch model loaded.")
except Exception as e:
    logging.error(f"❌ PyTorch model load error: {e}")
    model2 = None

# ImageNet Cloth Detector
imagenet_model = models.resnet18(pretrained=True)
imagenet_model.eval()
LABELS_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
imagenet_labels = [json.load(urllib.request.urlopen(LABELS_URL))[str(i)][1] for i in range(1000)]

imagenet_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ---------------- Class Labels ----------------
class_labels_model1 = ['Good', 'Objects', 'Hole', 'Oil Spot', 'Thread Error']
class_labels_model2 = ['vertical', 'defect-free', 'hole', 'horizontal', 'lines', 'stain']
CONFIDENCE_THRESHOLD = 0.6

# ---------------- Cloth Detection ----------------
def is_cloth_by_imagenet(img: Image.Image, allowed_keywords=[
    'suit', 'shirt', 'jean', 'tshirt', 'fabric', 'apparel',
    'sock', 'pajama', 'trouser', 'shorts', 'cloth', 'jacket',
    'sweater', 'dress', 'skirt', 'kurta', 'blazer', 'undergarment',
    'hoodie', 'vest', 'tracksuit', 'uniform','tick'
]):
    transformed = imagenet_transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = imagenet_model(tensor)
        probs = F.softmax(outputs[0], dim=0)
        top_classes = [imagenet_labels[i] for i in torch.topk(probs, 5).indices]
    logging.info(f"🧵 Top predictions: {top_classes}")
    return any(keyword in label.lower() for label in top_classes for keyword in allowed_keywords)



# ---------------- Image Preprocessing ----------------
# Define separate preprocessing for Keras model
keras_preprocess_transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to (64, 64) for Keras model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Define better preprocessing for PyTorch model
preprocess_transform = transforms.Compose([
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

def preprocess_image(image_data):
    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1]))).convert('RGB')
        img = enhance_edges(img)
        img_for_cloth_detection = img.copy()

        # For Keras model, resize to (64, 64)
        img_tensor_keras = keras_preprocess_transform(img)

        # For PyTorch model, resize to (224, 224)
        img_tensor_pytorch = preprocess_transform(img)

        return img_tensor_keras, img_tensor_pytorch, img_for_cloth_detection

    except Exception as e:
        raise ValueError("Invalid image data")

@image_bp.route('/predict_image', methods=['POST'])
def predict_image():
    if model1 is None or model2 is None:
        return jsonify({'error': 'Model(s) not loaded'}), 500
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400
        image_data = data['image']
        print(f"🖼️ Received image data: {image_data[:30]}...")  # Log first 30 characters for debugging
        img_tensor_keras, img_tensor_pytorch, img_pil = preprocess_image(image_data)
        
         # if not is_cloth_by_imagenet(img_pil):
        #     return jsonify({'error': 'Invalid image: no cloth detected'}), 400

        # --- Keras Prediction ---
        keras_input = img_tensor_keras.unsqueeze(0).numpy()
        keras_input = np.transpose(keras_input, (0, 2, 3, 1))  # (1, 64, 64, 3) -> (1, 64, 64, 3)

        prediction1 = model1.predict(keras_input)
        idx1, conf1 = np.argmax(prediction1), float(np.max(prediction1))

        # --- PyTorch Prediction ---
        torch_input = img_tensor_pytorch.unsqueeze(0)  # (1, 224, 224, 3)
        with torch.no_grad():
            prediction2 = model2(torch_input)
            
        outputs = F.softmax(prediction2, dim=1)
        confidence2, idx2 = torch.max(outputs, 1)
        label2 = class_labels_model2[idx2.item()]
        confidence2 = float(confidence2.item())
        

        result = {
            'model1': {'label': label1, 'confidence': confidence1} if confidence1 >= CONFIDENCE_THRESHOLD else None,
            'model2': {'label': label2, 'confidence': confidence2} if confidence2 >= CONFIDENCE_THRESHOLD else None
        }


        return jsonify({
            'model1': {'label': ['Good', 'Objects', 'Hole', 'Oil Spot', 'Thread Error'][idx1], 'confidence': conf1}
            if conf1 >= CONFIDENCE_THRESHOLD else None,
            'model2': {'label': ['vertical', 'defect', 'hole', 'horizontal', 'lines', 'stain'][idx2.item()], 'confidence': conf2.item()}
            if conf2.item() >= CONFIDENCE_THRESHOLD else None
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Server error: {e}")
        return jsonify({'error': 'Server error'}), 500
