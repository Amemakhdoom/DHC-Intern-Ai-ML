# ============================================
# Gradio Deployment — Customer Churn Predictor
# ============================================

import gradio as gr
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("churn_pipeline.pkl")

def predict_churn(gender, senior, partner, dependents, tenure,
                  phone, multiple, internet, online_sec, online_bak,
                  device, tech, tv, movies, contract, paperless,
                  payment, monthly, total):

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bak,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        return f"⚠️ LIKELY TO CHURN — Probability: {probability*100:.1f}%"
    else:
        return f"✅ NOT Likely to Churn — Probability: {probability*100:.1f}%"

# Gradio Interface
demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Dropdown(["Male", "Female"],                          label="Gender"),
        gr.Dropdown([0, 1],                                      label="Senior Citizen"),
        gr.Dropdown(["Yes", "No"],                               label="Partner"),
        gr.Dropdown(["Yes", "No"],                               label="Dependents"),
        gr.Slider(0, 72, value=12,                               label="Tenure (months)"),
        gr.Dropdown(["Yes", "No"],                               label="Phone Service"),
        gr.Dropdown(["Yes", "No", "No phone service"],           label="Multiple Lines"),
        gr.Dropdown(["DSL", "Fiber optic", "No"],                label="Internet Service"),
        gr.Dropdown(["Yes", "No", "No internet service"],        label="Online Security"),
        gr.Dropdown(["Yes", "No", "No internet service"],        label="Online Backup"),
        gr.Dropdown(["Yes", "No", "No internet service"],        label="Device Protection"),
        gr.Dropdown(["Yes", "No", "No internet service"],        label="Tech Support"),
        gr.Dropdown(["Yes", "No", "No internet service"],        label="Streaming TV"),
        gr.Dropdown(["Yes", "No", "No internet service"],        label="Streaming Movies"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"],  label="Contract"),
        gr.Dropdown(["Yes", "No"],                               label="Paperless Billing"),
        gr.Dropdown([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ],                                                        label="Payment Method"),
        gr.Number(value=65,                                       label="Monthly Charges ($)"),
        gr.Number(value=1500,                                     label="Total Charges ($)"),
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="📉 Customer Churn Predictor",
    description="Powered by Fine-Tuned Random Forest ML Pipeline | DevelopersHub Internship",
    examples=[[
        "Male", 0, "Yes", "No", 24,
        "Yes", "No", "DSL", "Yes", "No",
        "No", "Yes", "No", "No",
        "One year", "Yes", "Bank transfer (automatic)",
        55.0, 1400.0
    ]]
)

demo.launch(share=True)
