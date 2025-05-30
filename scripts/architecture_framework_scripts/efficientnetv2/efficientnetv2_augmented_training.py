import os
import random
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import cv2
from tqdm import tqdm
from sklearn.metrics import classification_report

# --- Config ---
DATASET_DIR = "dataset/PlantVillage_modified/Train"
KNN_MODEL_PATH = "scripts/architecture_framework_scripts/combination/glcm+knn_for_combination.pkl"
OUTPUT_MODEL_PATH = "scripts/architecture_framework_scripts/efficientnet_augmented1.pth"
NUM_CLASSES = 9
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_SPLIT = 0.3

# --- Preprocessing ---
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- GLCM Feature Extractor (6 features) ---
def extract_glcm_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0],
    ]
    return np.array(features).reshape(1, -1)

# --- Custom Dataset ---
class AugmentedPlantDataset(Dataset):
    def __init__(self, image_folder, knn_model):
        self.samples = image_folder.samples
        self.classes = image_folder.classes
        self.class_to_idx = image_folder.class_to_idx
        self.transform = image_transform
        self.knn = knn_model

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load image
        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)

        # Get KNN softmax predictions
        glcm_features = extract_glcm_features(path)
        knn_probs = self.knn.predict_proba(glcm_features).squeeze()  # Shape: (9,)

        return img_tensor, torch.tensor(knn_probs, dtype=torch.float32), label

# --- Augmented Model Definition ---
class AugmentedEfficientNet(nn.Module):
    def __init__(self, num_classes=9, knn_feature_dim=9):
        super().__init__()
        self.backbone = models.efficientnet_v2_s(weights=None)
        self.feature_dim = self.backbone.classifier[1].in_features  # <== Get it BEFORE replacing
        self.backbone.classifier = nn.Identity()  # Remove classifier


        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + knn_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, knn_vector):
        x_img = self.backbone(image)
        x = torch.cat((x_img, knn_vector), dim=1)
        out = self.classifier(x)
        return out

# --- Training Function ---
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0
    for imgs, knn_vecs, labels in tqdm(loader, desc="Training"):
        imgs = imgs.to(DEVICE)
        knn_vecs = knn_vecs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(imgs, knn_vecs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)

# --- Evaluation ---
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, knn_vecs, labels in tqdm(loader, desc="Validating"):
            imgs = imgs.to(DEVICE)
            knn_vecs = knn_vecs.to(DEVICE)
            outputs = model(imgs, knn_vecs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    report = classification_report(y_true, y_pred, output_dict=False)
    print(report)

# --- Main ---
if __name__ == "__main__":
    print("[INFO] Loading KNN model...")
    knn_model = joblib.load(KNN_MODEL_PATH)

    print("[INFO] Loading dataset and splitting...")
    full_dataset = ImageFolder(root=DATASET_DIR)
    dataset = AugmentedPlantDataset(full_dataset, knn_model)

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    print("[INFO] Initializing model...")
    model = AugmentedEfficientNet(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("[INFO] Starting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Train Loss: {train_loss:.4f}")
        evaluate(model, val_loader)

    print(f"[INFO] Saving model to {OUTPUT_MODEL_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
