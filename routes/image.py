from flask import Blueprint, request, jsonify
import tensorflow as tf
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

# Initialize Blueprint
image_bp = Blueprint('image', __name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# ---------------- Load Keras Model ----------------
try:
    model1 = tf.keras.models.load_model('./models/model_fold_3.h5', compile=False)
    logging.info("✅ Model 1 loaded successfully (Keras).")
except Exception as e:
    logging.error(f"❌ Error loading Model 1: {e}")
    model1 = None

# ---------------- Load PyTorch Model (ResNet18) ----------------
model2 = models.resnet18(pretrained=False)
num_classes = 6
model2.fc = nn.Linear(model2.fc.in_features, num_classes)

model_path = './models/mixeddataset_resnet_classweights.pth'
try:
    state_dict = torch.load(model_path, map_location='cpu')
    model2.load_state_dict(state_dict)
    model2.eval()
    logging.info("✅ Model 2 loaded successfully (PyTorch ResNet18).")
except Exception as e:
    logging.error(f"❌ Error loading Model 2: {e}")
    model2 = None

# ---------------- Load Pretrained ResNet18 for Cloth Detection ----------------
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
class_labels_model2 = ['vertical', 'defect', 'hole', 'horizontal', 'lines', 'stain']
CONFIDENCE_THRESHOLD = 0.6

# ---------------- Cloth Detection via ImageNet ----------------
def is_cloth_by_imagenet(img: Image.Image, allowed_keywords=['suit', 'shirt', 'jean', 'tshirt', 'fabric', 'apparel']):
    transformed = imagenet_transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = imagenet_model(transformed)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top5 = torch.topk(probs, 5)
        top_classes = [imagenet_labels[i] for i in top5.indices]

        logging.info(f"Top-5 ImageNet labels: {top_classes}")

        for label in top_classes:
            if any(keyword in label.lower() for keyword in allowed_keywords):
                return True
    return False

# ---------------- Preprocess Image ----------------
def preprocess_image(image_data):
    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
        img = img.convert('RGB')
        img_resized = img.resize((64, 64))
        img_array = np.array(img_resized) / 255.0
        return img_array, img
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        raise ValueError("Error processing the image.")

# ---------------- Prediction Endpoint ----------------
@image_bp.route('/predict_image', methods=['POST'])
def predict_image():
    if model1 is None or model2 is None:
        return jsonify({'error': 'One or both models failed to load'}), 500

    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400

        image_data = data['image']
        img_array, img_pil = preprocess_image(image_data)

        # Check if it's a cloth image
        if not is_cloth_by_imagenet(img_pil):
            return jsonify({'error': 'Wrong image - cloth not detected'}), 400

        # --- Model 1 Prediction (Keras) ---
        img_array_keras = np.expand_dims(img_array, axis=0)
        prediction1 = model1.predict(img_array_keras)
        predicted_index1 = np.argmax(prediction1)
        predicted_class1 = class_labels_model1[predicted_index1]
        confidence1 = prediction1[0][predicted_index1]

        # --- Model 2 Prediction (PyTorch) ---
        img_tensor = torch.tensor(img_array.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction2 = model2(img_tensor)
        confidence2, predicted_index2 = torch.max(prediction2, 1)
        predicted_class2 = class_labels_model2[predicted_index2.item()]
        confidence2 = confidence2.item()

        logging.info(f"Model 1 Prediction: {predicted_class1} ({confidence1:.2f})")
        logging.info(f"Model 2 Prediction: {predicted_class2} ({confidence2:.2f})")

        result = {
            'model1': {'label': predicted_class1, 'confidence': float(confidence1)} if confidence1 >= CONFIDENCE_THRESHOLD else None,
            'model2': {'label': predicted_class2, 'confidence': float(confidence2)} if confidence2 >= CONFIDENCE_THRESHOLD else None,
        }

        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500
