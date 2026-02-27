import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import gradio as gr
import numpy as np
import joblib
from PIL import Image as PILImage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultimodalHousingModel(nn.Module):
    def __init__(self, tabular_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.cnn_fc   = nn.Linear(128 * 8 * 8, 256)
        self.tab_fc   = nn.Sequential(
            nn.Linear(tabular_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
        )
        self.combined = nn.Sequential(
            nn.Linear(256 + 128, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    def forward(self, img, tabular):
        x_img = self.cnn(img)
        x_img = x_img.view(x_img.size(0), -1)
        x_img = F.relu(self.cnn_fc(x_img))
        x_tab = self.tab_fc(tabular)
        x     = torch.cat([x_img, x_tab], dim=1)
        return self.combined(x).squeeze(1)

model  = MultimodalHousingModel(tabular_dim=8).to(device)
model.load_state_dict(torch.load("housing_model.pth", map_location=device))
model.eval()
scaler = joblib.load("scaler.pkl")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def generate_house_image(features):
    img_array  = np.zeros((64, 64, 3), dtype=np.uint8)
    rooms      = min(int(abs(features[2])), 8)
    age        = min(int(abs(features[3]) / 10), 6)
    income     = min(int(abs(features[0])), 7)
    bg_color   = max(0, min(255, int((income / 8) * 200) + 55))
    bg_color2  = max(0, min(255, bg_color - 20))
    bg_color3  = max(0, min(255, bg_color - 40))
    img_array[:, :] = [bg_color, bg_color2, bg_color3]
    img_array[20:55, 10:54] = [180, 140, 100]
    roof_val   = max(0, min(255, 150 - age * 10))
    for i in range(15):
        img_array[5 + i, 10 + i:54 - i] = [roof_val, roof_val, roof_val]
    for r in range(min(rooms, 4)):
        x = 12 + r * 10
        img_array[28:38, x:x+6] = [100, 180, 255]
    return PILImage.fromarray(img_array)

def predict_price(image, med_inc, house_age, ave_rooms, ave_bedrms,
                  population, ave_occup, latitude, longitude):
    input_tab  = scaler.transform([[
        med_inc, house_age, ave_rooms, ave_bedrms,
        population, ave_occup, latitude, longitude
    ]])
    tab_tensor = torch.FloatTensor(input_tab).to(device)
    if image is not None:
        img = PILImage.fromarray(image).resize((64, 64))
    else:
        img = generate_house_image(input_tab[0])
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img_tensor, tab_tensor).item()
    pred = max(0, pred)
    return f"Predicted House Price: USD {pred * 100000:,.0f}"

demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Image(label="Upload House Image (optional)",
                 sources=["upload", "webcam", "clipboard"]),
        gr.Slider(0.5, 15.0,  value=3.0,    label="Median Income (10k)"),
        gr.Slider(1,   52,    value=20,     label="House Age (years)"),
        gr.Slider(1,   20,    value=5.0,    label="Average Rooms"),
        gr.Slider(1,   5,     value=1.0,    label="Average Bedrooms"),
        gr.Slider(100, 5000,  value=1000,   label="Population"),
        gr.Slider(1,   10,    value=3.0,    label="Average Occupants"),
        gr.Slider(32,  42,    value=37.0,   label="Latitude"),
        gr.Slider(-125,-114,  value=-120.0, label="Longitude"),
    ],
    outputs=gr.Textbox(label="Price Prediction"),
    title="Housing Price Predictor",
    description="Multimodal ML - CNN + Tabular | DevelopersHub Internship",
)
demo.launch(share=True)
