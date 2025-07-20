# 🗑️ Trash Detection using Deep Learning

This repository contains an end-to-end pipeline for detecting trash (e.g., plastic, metal, paper, etc.) using deep learning techniques. The model is based on a CNN architecture (e.g., ResNet-50) and can be used to classify and detect garbage for real-world sustainability applications.

## 📌 Project Highlights

- 🔍 Trained a **ResNet-50** model with an accuracy of **93%**
- 📂 Dataset annotated using **Labelbox**
- 🌍 Street-level images sourced from **Mapillary**
- 🔁 Includes data preprocessing, training, validation, and inference pipeline
- 🧪 Clean modular structure for easy experimentation and deployment

---

## 📁 Folder Structure
```plaintext
trash-detection/
├── data/             # Raw and preprocessed data
├── labels/           # Annotations from Labelbox
├── model/            # Trained model files (.pth/.pt)
├── notebook/         # Jupyter notebooks for EDA, training, etc.
├── src/              # Source code: training, inference, utils
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
