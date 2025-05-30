import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from torchvision import transforms, models
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
VAL_DIR = "dataset/PlantVillage_modified/Train"
KNN_MODEL_PATH = "scripts/architecture_framework_scripts/combination/glcm+knn_for_combination.pkl"
MODEL_PATH = "scripts/architecture_framework_scripts/efficientnet_augmented.pth"
NUM_CLASSES = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = [
    "Bacterial Spot", "Cercospora Leaf Spot", "Common Rust", "Early Blight",
    "Healthy", "Late Blight", "Northern Leaf Blight",
    "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- GLCM extractor ---
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

# --- Model ---
class AugmentedEfficientNet(nn.Module):
    def __init__(self, num_classes=9, knn_feature_dim=9):
        super().__init__()
        self.backbone = models.efficientnet_v2_s(weights=None)
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + knn_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, knn_vector):
        x_img = self.backbone(image)
        x = torch.cat((x_img, knn_vector), dim=1)
        return self.classifier(x)

# --- Loaders ---
def load_model():
    model = AugmentedEfficientNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()

# --- Inference on folder ---
def run_inference(val_dir, model, knn):
    all_images = []
    y_true = []
    y_pred = []

    for class_name in os.listdir(val_dir):
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_dir): continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")): continue
            img_path = os.path.join(class_dir, fname)
            try:
                # image
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)

                # knn prob
                glcm_feat = extract_glcm_features(img_path)
                knn_probs = knn.predict_proba(glcm_feat)
                knn_tensor = torch.tensor(knn_probs, dtype=torch.float32).to(DEVICE)

                # predict
                with torch.no_grad():
                    output = model(img_tensor, knn_tensor)
                    pred = output.argmax(dim=1).item()

                all_images.append(fname)
                y_true.append(CLASS_TO_IDX[class_name])
                y_pred.append(pred)

            except Exception as e:
                print(f"[⚠️] Failed on {img_path}: {e}")

    return all_images, y_true, y_pred

# --- Save results ---
def save_results(imgs, y_true, y_pred, output_dir="results/augmented_evalpd"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame({
        "Image": imgs,
        "True_Label": [CLASS_NAMES[i] for i in y_true],
        "Predicted_Label": [CLASS_NAMES[i] for i in y_pred]
    })
    df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(os.path.join(output_dir, "classification_report.csv"))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Augmented EfficientNet Inference - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    print(f"[✅] Results saved to {output_dir}")

# --- Main ---
if __name__ == "__main__":
    print("[INFO] Loading models...")
    knn_model = joblib.load(KNN_MODEL_PATH)
    model = load_model()

    print("[INFO] Running inference...")
    imgs, y_true, y_pred = run_inference(VAL_DIR, model, knn_model)

    print("[INFO] Saving results...")
    save_results(imgs, y_true, y_pred)
