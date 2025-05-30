import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import cv2
from skimage.feature import graycomatrix, graycoprops

# --- Config ---
ARCH_NAME = "convnext"
NUM_CLASSES = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "Bacterial Spot", "Cercospora Leaf Spot", "Common Rust", "Early Blight",
    "Healthy", "Late Blight", "Northern Leaf Blight",
    "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]
KNN_MODEL_PATH = "models/glcm_knn/trained/glcm_knn.pkl"
MODEL_PATH = "models/convnext/augmented/convnext_augmented.pth"
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- GLCM Feature Extractor ---
def extract_glcm_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded.")
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

# --- Augmented ConvNeXt Model ---
class AugmentedConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.backbone.classifier[2] = nn.Identity()

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

# --- Inference ---
def run_augmented_inference(dataset_path, dataset_name):
    print(f"[INFO] Running augmented inference on {dataset_name}...")
    dataset = datasets.ImageFolder(dataset_path)
    knn = joblib.load(KNN_MODEL_PATH)
    model = AugmentedConvNeXt().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []

    for path, label in tqdm(dataset.samples):
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
            glcm_feat = extract_glcm_features(path).reshape(1, -1)
            knn_probs = torch.tensor(knn.predict_proba(glcm_feat), dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                output = model(img_tensor, knn_probs)
                pred = torch.argmax(output, dim=1).item()
            y_true.append(label)
            y_pred.append(pred)
        except Exception as e:
            print(f"[WARN] Skipped {path}: {e}")

    out_dir = f"results/{ARCH_NAME}/augmented/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)

    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(out_dir, f"{ARCH_NAME}_inference_result.csv"), index=True)
    with open(os.path.join(out_dir, f"{ARCH_NAME}_inference_result.txt"), "w") as f:
        f.write(report_df.to_string())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{ARCH_NAME.replace('_', ' ').title()} Augmented Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{ARCH_NAME}_confusion_matrix.png"))
    plt.close()

    print(f"[✅] Results saved to: {out_dir}")

# --- Main ---
if __name__ == "__main__":
    run_augmented_inference("dataset/plantvillage/test", "plantvillage")
    run_augmented_inference("dataset/plantdoc/test", "plantdoc")
