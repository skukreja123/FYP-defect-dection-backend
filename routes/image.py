from flask import Blueprint, request, jsonify
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from config import Config
from Utils.JWTtoken import token_required
from models.Frame import insert_frame_with_predictions , get_frame_by_id, get_all_frames_by_user_id, delete_frame_by_id

# Load environment variables
load_dotenv()

# Initialize Blueprint
image_bp = Blueprint('image', __name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# ---------------- Model Download from Google Drive ----------------
MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

pthid = Config.GDRIVE_MODEL_ID
h5id = Config.h5py_model_ID

pth_model_path = os.path.join(MODEL_DIR, 'resnet_model.pth')
h5_model_path = os.path.join(MODEL_DIR, 'keras_model.h5')

if not pthid or not h5id:
    raise EnvironmentError("❌ Google Drive model IDs not set in .env")

if not os.path.exists(pth_model_path):
    logging.info("⬇️ Downloading PyTorch model...")
    gdown.download(f"https://drive.google.com/uc?id={pthid}", pth_model_path, quiet=False)

if not os.path.exists(h5_model_path):
    logging.info("⬇️ Downloading Keras model...")
    gdown.download(f"https://drive.google.com/uc?id={h5id}", h5_model_path, quiet=False)

# ---------------- Load Keras Model ----------------
try:
    model1 = tf.keras.models.load_model(h5_model_path, compile=False)
    logging.info("✅ Keras model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading Keras model: {e}")
    model1 = None

# ---------------- Load PyTorch Model ----------------
model2 = models.resnet18(pretrained=False)
model2.fc = nn.Linear(model2.fc.in_features, 6)


try:
    state_dict = torch.load(pth_model_path, map_location='cpu')
    model2.load_state_dict(state_dict)
    model2.eval()
    logging.info("✅ PyTorch model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading PyTorch model: {e}")
    model2 = None

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
        outputs = imagenet_model(transformed)
        probs = F.softmax(outputs[0], dim=0)
        top5 = torch.topk(probs, 5)
        top_classes = [imagenet_labels[i] for i in top5.indices]

        logging.info(f"🧵 Top-5 ImageNet labels: {top_classes}")

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
        logging.error(f"❌ Error during image preprocessing: {e}")
        raise ValueError("Error processing the image.")

# ---------------- Prediction Route ----------------
@image_bp.route('/predict_image', methods=['POST'])
@token_required
def predict_image(user_id):
    # print(f"🔑 User ID: {user_id}")  # Log user ID for debugging
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
        idx1 = np.argmax(prediction1)
        label1 = class_labels_model1[idx1]
        confidence1 = float(prediction1[0][idx1])

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
                # Assuming you have `user_id` from the token or request context
        frame_bytes = base64.b64decode(image_data.split(',')[1])
        frame_id = insert_frame_with_predictions(
            user_id=user_id,
            frame_data=frame_bytes,
            keras_label=label1 if confidence1 >= CONFIDENCE_THRESHOLD else None,
            keras_confidence=confidence1 if confidence1 >= CONFIDENCE_THRESHOLD else None,
            pytorch_label=label2 if confidence2 >= CONFIDENCE_THRESHOLD else None,
            pytorch_confidence=confidence2 if confidence2 >= CONFIDENCE_THRESHOLD else None,
        )


        return jsonify(result)

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logging.error(f"❌ Internal error: {e}")
        return jsonify({'error': 'Internal server error'}), 500




@image_bp.route('/<int:frame_id>', methods=['GET'])
@token_required
def get_image(frame_id):
    frame = get_frame_by_id(frame_id)
    
    if frame is None:
        return jsonify({'error': 'Frame not found'}), 404

    frame_data = frame[2]  # assuming frame_data is the third column (index 2)

    encoded_img = base64.b64encode(frame_data).decode('utf-8')
    return jsonify({
        'image_base64': f"data:image/jpeg;base64,{encoded_img}",
        'keras_label': frame[3],
        'keras_confidence': frame[4],
        'pytorch_label': frame[5],
        'pytorch_confidence': frame[6]
    })
    
    
@image_bp.route('/All_frame', methods=['GET'])
@token_required
def get_all_frames(user_id):
    all_frames = get_all_frames_by_user_id(user_id)
    if all_frames is None:
         return jsonify({'error': 'No frames found for this user'}), 404
     
    all_frames_data = []
    for frame in all_frames:
        frame_data = frame[2]
        encoded_img = base64.b64encode(frame_data).decode('utf-8')
        all_frames_data.append({
            'frame_id': frame[0],
            'image_base64': f"data:image/jpeg;base64,{encoded_img}",
            'keras_label': frame[3],
            'keras_confidence': frame[4],
            'pytorch_label': frame[5],
            'pytorch_confidence': frame[6]
        })
    return jsonify(all_frames_data), 200


@image_bp.route('/delete_frame/<int:frame_id>', methods=['DELETE'])
def delete_frame(frame_id):
    try:
        print(f"🗑️ Deleting frame with ID: {frame_id}")  # Log frame ID for debugging
        delete_frame_by_id(frame_id)
        return jsonify({'message': 'Frame deleted successfully'}), 200
    except Exception as e:
        logging.error(f"❌ Error deleting frame: {e}")
        return jsonify({'error': 'Error deleting frame'}), 500
    