# Fabric Defect Detection – Backend API

This repository contains the backend API for a **real-time fabric defect detection system**, developed using **Flask** and powered by a fine-tuned **ResNet deep learning model**. The system helps textile manufacturers automate the visual inspection of fabrics to identify defects such as holes, stains, and pattern irregularities.

It pairs with a React-based frontend, delivering a full-stack solution for industrial-scale fabric quality control.

---

## 🚀 Features

- 🎯 **ResNet-Based Classification:** Robust classification of fabric defects using a fine-tuned ResNet model trained on real textile datasets.
- 🧠 **Deep Learning Integration:** Loads a `.pth` PyTorch model for defect detection and classification.
- 🔗 **REST API Ready:** Clean endpoints for image upload and retrieving prediction results.
- 🗃️ **PostgreSQL Database:** Logs predictions, images, and timestamps via SQLAlchemy ORM.
- ⚡ **Frontend Integration:** Works seamlessly with the React frontend (link below).
- 📁 **Modular & Scalable:** Designed for easy maintenance and future enhancements.

---

## 🧠 Model & Datasets

- **Pre-trained Model:** `model.pth` (place inside `./model/` folder) — used for making predictions.
- **Datasets Used for Training:**
  - [Fabric Defect Dataset – Kaggle](https://www.kaggle.com/datasets)
  - [TILDA 400 Patches – Kaggle](https://www.kaggle.com/datasets)

---

## 📁 Project Structure


FYP-defect-detection-backend/
├── app.py # Flask app entry point
├── model/
│ └── model.pth # Trained ResNet model
├── database/
│ └── models.py # SQLAlchemy models for PostgreSQL
├── utils/
│ ├── preprocessing.py # Image preprocessing functions
│ └── predict.py # Model loading and prediction logic
├── static/uploads/ # Uploaded images directory
├── config.py # Configurations (DB URI, env variables)
├── requirements.txt # Python dependencies
└── README.md # Project documentation



---

## ⚙️ Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/skukreja123/FYP-defect-dection-backend.git
   cd FYP-defect-dection-backend

Set up a virtual environment:

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt

Configure PostgreSQL:

Edit config.py and update the database URI:

DATABASE_URI = 'postgresql://username:password@localhost/fabric_defects'


Then create the database and run migrations if necessary.

Run the Flask server:


python app.py

The API will be available at http://127.0.0.1:5000/.

🧪 API Endpoints
POST /upload
Description: Upload a fabric image for defect prediction.

Form Data:

image — JPEG or PNG file.

Response:
{
  "prediction": "Hole",
  "confidence": 0.94
}


GET /results
Description: Retrieve the history of logged predictions (if implemented).

Response: JSON list of predictions.

🛠 Technologies Used

Layer	Tools
Backend API	Flask , Flask-CORS
Deep Learning	PyTorch, ResNet
Database	PostgreSQL, SQLAlchemy
Dev Tools	Postman, Git, Virtualenv


📸 Sample Prediction Output:

![image](https://github.com/user-attachments/assets/590ced5e-dc72-4668-bc0a-6d3dcb31ed96)



🤝 Contributors
Sahil Kukreja – Developer, Model Trainer, Backend Engineer

Areeb – Developer, Model Trainer, Backend Engineer

Mustafa – Developer, Model Trainer, Backend Engineer

GitHub Profile

🔗 Related Repositories
Frontend: React Fabric Defect Detection Frontend (https://github.com/skukreja123/FYP-defect-detection-frontend)
