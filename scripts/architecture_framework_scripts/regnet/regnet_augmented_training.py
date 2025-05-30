import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, Dataset
import joblib
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split

# --- Config ---
DATASET_PATH = "dataset/augmented/test"
KNN_MODEL_PATH = "models/glcm_knn/trained/glcm_knn.pkl"
MODEL_SAVE_PATH = "models/regnet/augmented/regnet_best_model.pth"
NUM_CLASSES = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_EPOCHS = 10

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- GLCM Feature Extractor ---
def extract_glcm_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ])

# --- Augmented Dataset ---
class AugmentedDataset(Dataset):
    def __init__(self, image_folder, knn_model):
        self.samples = image_folder.samples
        self.transform = TRANSFORM
        self.knn = knn_model

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)

        glcm_feat = extract_glcm_features(path).reshape(1, -1)
        knn_probs = self.knn.predict_proba(glcm_feat).flatten()

        return img_tensor, torch.tensor(knn_probs, dtype=torch.float32), label

# --- Augmented Model ---
class AugmentedRegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()

        # Get output size dynamically
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feature_dim = self.backbone(dummy).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + NUM_CLASSES, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, img_tensor, knn_tensor):
        features = self.backbone(img_tensor)
        combined = torch.cat((features, knn_tensor), dim=1)
        return self.classifier(combined)

# --- Train Function ---
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0
    for img_tensor, knn_tensor, label in tqdm(dataloader, desc="Training"):
        img_tensor, knn_tensor, label = img_tensor.to(DEVICE), knn_tensor.to(DEVICE), label.to(DEVICE)
        output = model(img_tensor, knn_tensor)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)

# --- Main ---
if __name__ == "__main__":
    print("[INFO] Loading KNN model...")
    knn_model = joblib.load(KNN_MODEL_PATH)

    print("[INFO] Preparing dataset...")
    base_dataset = datasets.ImageFolder(DATASET_PATH)
    full_dataset = AugmentedDataset(base_dataset, knn_model)

    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print("[INFO] Initializing model...")
    model = AugmentedRegNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Training Loss: {loss:.4f}")

    print(f"[INFO] Saving model to {MODEL_SAVE_PATH}")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("[✅] Augmented RegNet model saved.")
