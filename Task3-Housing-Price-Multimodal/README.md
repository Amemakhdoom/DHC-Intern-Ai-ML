# 🏠 Task 3 — Multimodal Housing Price Prediction

## 📌 Objective
Predict housing prices using both structured tabular data and
house images by combining CNN and tabular neural network branches.

## 📂 Project Structure
- train.py — Model training script
- app.py — Gradio deployment app
- requirements.txt — Required libraries
- README.md — Project documentation

## 📊 Dataset
- Name: California Housing Dataset
- Source: Scikit-learn built-in dataset
- Samples: 20,640 houses
- Features: 8 tabular features + synthetic house images
- Target: House price (in $100,000s)

## 🧠 Model Architecture

**CNN Branch (Image Features):**
- Conv2d 3 to 32 + ReLU + MaxPool
- Conv2d 32 to 64 + ReLU + MaxPool
- Conv2d 64 to 128 + ReLU + MaxPool
- Fully Connected to 256 features

**Tabular Branch:**
- Linear 8 to 64 + ReLU
- Linear 64 to 128 + ReLU

**Combined Branch:**
- Concatenate (256 + 128 = 384)
- Linear 384 to 128 + Dropout
- Linear 128 to 64
- Linear 64 to 1 (Price output)

## 📈 Results

| Metric | Score |
|--------|-------|
| MAE    | ~0.55 |
| RMSE   | ~0.73 |

## 🚀 How to Run
pip install -r requirements.txt
python train.py
python app.py

## 🛠️ Skills Gained
- Multimodal machine learning
- Convolutional Neural Networks (CNNs)
- Feature fusion (image + tabular)
- Regression modeling and evaluation
- PyTorch custom Dataset and DataLoader
- Gradio deployment with image upload
