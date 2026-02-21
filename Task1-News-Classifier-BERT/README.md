# 📰 Task 1 — News Topic Classifier Using BERT

## 📌 Objective
Fine-tune a BERT transformer model to classify news headlines
into 4 topic categories using the AG News dataset.

## 📂 Project Structure
- train.py — Training and fine-tuning script
- app.py — Gradio deployment app
- requirements.txt — Required libraries
- README.md — Project documentation

## 📊 Dataset
- Name: AG News
- Source: Hugging Face Datasets
- Classes: World | Sports | Business | Sci/Tech
- Train samples: 120,000
- Test samples: 7,600

## 🧠 Model
- Base Model: bert-base-uncased
- Task: Sequence Classification
- Classes: 4
- Epochs: 3
- Learning Rate: 2e-5

## 📈 Results
| Metric   | Score |
|----------|-------|
| Accuracy | ~93%  |
| F1 Score | ~0.93 |

## 🚀 How to Run
pip install -r requirements.txt
python train.py
python app.py

## 🛠️ Skills Gained
- NLP using Transformers
- Transfer learning and fine-tuning
- Evaluation metrics for text classification
- Lightweight model deployment with Gradio
