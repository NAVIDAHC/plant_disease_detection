import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from skimage.feature import graycomatrix, graycoprops
import joblib

# --- Config ---
ARCH_NAME = "resnet50"
NUM_CLASSES = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "Bacterial Spot", "Cercospora Leaf Spot", "Common Rust", "Early Blight",
    "Healthy", "Late Blight", "Northern Leaf Blight",
    "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]
DL_MODEL_PATH = "models/resnet50/augmented/resnet50_augmented_model.pth"
KNN_MODEL_PATH = "models/glcm_knn/trained/glcm_knn.pkl"
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Model ---
class AugmentedResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.feature_dim = 2048
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + NUM_CLASSES, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, img_tensor, knn_vector):
        x = self.backbone(img_tensor)
        combined = torch.cat((x, knn_vector), dim=1)
        return self.classifier(combined)

# --- GLCM ---
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
        graycoprops(glcm, 'ASM')[0, 0],
    ]).reshape(1, -1)

# --- Dataset Walker ---
def get_image_paths_and_labels(root_dir):
    image_paths, labels = [], []
    class_to_idx = {cls: i for i, cls in enumerate(sorted(os.listdir(root_dir)))}
    for cls in sorted(os.listdir(root_dir)):
        cls_dir = os.path.join(root_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls])
    return image_paths, labels

# --- Inference + Save ---
def run_augmented_inference(dataset_path, dataset_name):
    print(f"[INFO] Running augmented inference on {dataset_name}...")
    image_paths, labels = get_image_paths_and_labels(dataset_path)
    model = AugmentedResNet50()
    model.load_state_dict(torch.load(DL_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    knn = joblib.load(KNN_MODEL_PATH)
    y_true, y_pred = [], []

    for path, true_label in tqdm(zip(image_paths, labels), total=len(labels)):
        try:
            glcm_feat = extract_glcm_features(path)
            knn_probs = knn.predict_proba(glcm_feat).flatten()

            img = Image.open(path).convert("RGB")
            input_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
            knn_tensor = torch.tensor(knn_probs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor, knn_tensor)
                pred = torch.argmax(output, dim=1).item()

            y_true.append(true_label)
            y_pred.append(pred)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")

    # Save results
    out_dir = f"results/resnet50/augmented/{dataset_name}"
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
    plt.title(f"{ARCH_NAME.upper()} Augmented Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{ARCH_NAME}_confusion_matrix.png"))
    plt.close()

    print(f"[✅] Results saved to: {out_dir}")

# --- Main ---
if __name__ == "__main__":
    run_augmented_inference("dataset/plantvillage/test", "plantvillage")
    run_augmented_inference("dataset/plantdoc/test", "plantdoc")