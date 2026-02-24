# 📉 Task 2 — End-to-End ML Pipeline for Customer Churn Prediction

## 📌 Objective
Build a reusable and production-ready machine learning pipeline
to predict customer churn using the Telco Churn Dataset.

## 📂 Project Structure
```
Task2-ML-Pipeline-Churn/
├── train.py           # Training, pipeline building, evaluation
├── app.py             # Gradio deployment app
├── requirements.txt   # Required libraries
└── README.md          # Project documentation
```

## 📊 Dataset
- **Name:** Telco Customer Churn Dataset
- **Source:** IBM / GitHub
- **Rows:** 7,043 customers
- **Target:** Churn (Yes / No)
- **Features:** 20 (tenure, contract type, monthly charges, etc.)

## 🔄 Methodology

### 1. Data Preprocessing
- Dropped irrelevant columns (customerID)
- Fixed TotalCharges column (converted to numeric)
- Encoded target variable (Yes→1, No→0)
- Applied StandardScaler on numerical features
- Applied OneHotEncoder on categorical features

### 2. Pipeline Construction
Used Scikit-learn Pipeline API with ColumnTransformer:
- Numerical → StandardScaler
- Categorical → OneHotEncoder
- Model → LogisticRegression / RandomForestClassifier

### 3. Hyperparameter Tuning
Used GridSearchCV with 5-fold cross validation:
- n_estimators: [100, 200]
- max_depth: [5, 10, None]
- min_samples_split: [2, 5]

### 4. Model Export
Exported best pipeline using joblib for production use.

## 📈 Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | ~80% | ~0.58 |
| Random Forest | ~79% | ~0.55 |
| Random Forest + GridSearchCV | ~82% | ~0.62 |

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train and export model
```bash
python train.py
```

### Launch Gradio app
```bash
python app.py
```

## 🛠️ Skills Gained
- ML pipeline construction with Scikit-learn
- Hyperparameter tuning with GridSearchCV
- Model export and reusability with joblib
- Production-readiness practices
- Gradio deployment

## 📦 Libraries Used
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Joblib
- Gradio
