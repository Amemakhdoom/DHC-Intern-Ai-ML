import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# 1. Load Dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target
print("Dataset loaded")

# 2. Generate House Images
def generate_house_image(features):
    img_array = np.zeros((64, 64, 3), dtype=np.uint8)
    rooms     = min(int(abs(features[2])), 8)
    age       = min(int(abs(features[3]) / 10), 6)
    income    = min(int(abs(features[0])), 7)
    bg        = max(0, min(255, int((income / 8) * 200) + 55))
    bg2       = max(0, min(255, bg - 20))
    bg3       = max(0, min(255, bg - 40))
    img_array[:, :] = [bg, bg2, bg3]
    img_array[20:55, 10:54] = [180, 140, 100]
    roof_val  = max(0, min(255, 150 - age * 10))
    for i in range(15):
        img_array[5 + i, 10 + i:54 - i] = [roof_val, roof_val, roof_val]
    for r in range(min(rooms, 4)):
        x = 12 + r * 10
        img_array[28:38, x:x+6] = [100, 180, 255]
    return Image.fromarray(img_array)

# 3. Dataset Class
class HousingDataset(Dataset):
    def __init__(self, tabular_data, targets, transform=None):
        self.tabular     = torch.FloatTensor(tabular_data)
        self.targets     = torch.FloatTensor(targets)
        self.transform   = transform
        self.raw_tabular = tabular_data
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        img = generate_house_image(self.raw_tabular[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.tabular[idx], self.targets[idx]

# 4. Preprocessing
scaler   = StandardScaler()
X        = scaler.fit_transform(df.drop(columns=["Price"]).values)
y        = df["Price"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = HousingDataset(X_train, y_train, transform)
test_dataset  = HousingDataset(X_test,  y_test,  transform)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# 5. Model
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
            nn.Linear(64, 128),         nn.ReLU(),
        )
        self.combined = nn.Sequential(
            nn.Linear(256 + 128, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, img, tabular):
        x_img = self.cnn(img)
        x_img = x_img.view(x_img.size(0), -1)
        x_img = F.relu(self.cnn_fc(x_img))
        x_tab = self.tab_fc(tabular)
        x     = torch.cat([x_img, x_tab], dim=1)
        return self.combined(x).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MultimodalHousingModel(tabular_dim=X_train.shape[1]).to(device)
print("Model ready on:", device)

# 6. Train
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
EPOCHS    = 15

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, tabs, targets in train_loader:
        imgs, tabs, targets = imgs.to(device), tabs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs, tabs), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    scheduler.step()
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f}")

# 7. Evaluate
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for imgs, tabs, targets in test_loader:
        imgs, tabs = imgs.to(device), tabs.to(device)
        all_preds.extend(model(imgs, tabs).cpu().numpy())
        all_targets.extend(targets.numpy())

mae  = mean_absolute_error(all_targets, all_preds)
rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

# 8. Save
torch.save(model.state_dict(), "housing_model.pth")
joblib.dump(scaler, "scaler.pkl")
print("Model saved")
