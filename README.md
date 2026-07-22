# AI-ML Internship Project

### Cancer Detection • Face Recognition • Wildlife Detection

A comprehensive collection of three advanced Computer Vision and Deep Learning projects, carefully curated and rebranded for internship demonstration, academic portfolio, and practical learning purposes.

This repository showcases end-to-end AI solutions covering **medical imaging**, **biometric recognition**, and **ecological monitoring** — three high-impact domains in modern Artificial Intelligence.

---

## 📌 Table of Contents
- [About The Project](#about-the-project)
- [Projects Overview](#projects-overview)
  - [1. Cancer Detection](#1-cancer-detection)
  - [2. Face Recognition](#2-face-recognition)
  - [3. Wildlife Detection](#3-wildlife-detection)
- [Repository Structure](#repository-structure)
- [Customer Churn Assignment](#customer-churn-assignment)

---

## 📖 About The Project

This repository was created as part of an **AI/ML Internship Project**. It combines three open-source deep learning frameworks into a single, well-organized monorepo.

The goal is to demonstrate:
- Ability to work with large, real-world open-source AI codebases
- Understanding of medical AI, biometric systems, and computer vision for conservation
- Skills in project organization, documentation, and rebranding
- Practical knowledge of PyTorch-based deep learning pipelines

Each project has been lightly modified (mainly documentation and presentation) while preserving the original high-quality code, licenses, and functionality.

---

## 🔍 Projects Overview

### 1. Cancer Detection
**Folder:** `cancer-detection`  
**Based on:** [Project MONAI](https://github.com/Project-MONAI/MONAI)

MONAI (Medical Open Network for AI) is a PyTorch-based, open-source framework designed for deep learning in healthcare imaging. This module focuses on tumor segmentation, classification, and medical image analysis.

### 2. Face Recognition
**Folder:** `face-recognition`  
**Based on:** [InsightFace](https://github.com/deepinsight/insightface)

InsightFace provides state-of-the-art models for face detection, face recognition, alignment, and attribute analysis.

### 3. Wildlife Detection
**Folder:** `wildlife-detection`  
**Based on:** [Microsoft CameraTraps](https://github.com/microsoft/CameraTraps)

This project uses Microsoft’s MegaDetector and CameraTraps toolkit for wildlife monitoring with camera trap images.

---

## 📁 Repository Structure

```
AI-ML-Internship-Project/
│
├── README.md
├── cancer-detection/
├── face-recognition/
└── wildlife-detection/
```

---

## Customer Churn Assignment

### Objective
Predict whether a telecom customer is likely to churn using demographic and service-usage information from the Telco Customer Churn dataset.

### Dataset Link
- [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Files Added
- `Assignment-2.py` - runnable solution for all assignment tasks
- `Assignment-2.ipynb` - notebook version of the assignment
- `requirements.txt` - Python dependencies

### Libraries Used
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `kagglehub`

### Methodology
1. Load the dataset with Pandas and inspect the first five rows.
2. Identify numerical features, categorical features, and the target variable (`Churn`).
3. Check missing values and handle them with imputation.
4. Encode categorical features with one-hot encoding and split the data into 80% training and 20% testing.
5. Train a Logistic Regression model using a preprocessing pipeline.
6. Evaluate the model with Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

### Results
The script can load the CSV from this folder or download it automatically with `kagglehub`.

### Conclusion
Logistic Regression is a strong baseline for churn prediction because it is simple, fast, and easy to interpret. In this problem, churn is typically influenced by factors such as tenure, contract type, monthly charges, and service usage patterns. A limitation of Logistic Regression is that it assumes a largely linear relationship between features and the target, so it may miss complex nonlinear interactions in customer behavior.

### Note
Do not upload the dataset to GitHub. Keep only the code and documentation in the repository, and include the Kaggle link instead.
