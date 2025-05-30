import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops

# --- Config ---
DATASET_PATH = "dataset/plantdoc/val"
KNN_MODEL_PATH = "models/glcm_knn/trained/glcm_knn.pkl"
DL_MODEL_PATH = "models/regnet/trained/regnet_best_model.pth"
OUTPUT_META_MODEL_PATH = "models/regnet/meta_learner/regnet_meta_learner.pkl"
NUM_CLASSES = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "Bacterial Spot", "Cercospora Leaf Spot", "Common Rust", "Early Blight",
    "Healthy", "Late Blight", "Northern Leaf Blight",
    "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- GLCM Extractor ---
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
    ]).reshape(1, -1)

# --- RegNet Loader ---
def load_regnet_model():
    model = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(DL_MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()

# --- Image Path Loader ---
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

# --- Main ---
if __name__ == "__main__":
    print("[INFO] Loading models...")
    knn = joblib.load(KNN_MODEL_PATH)
    model = load_regnet_model()

    print("[INFO] Extracting stacking features...")
    image_paths, labels = get_image_paths_and_labels(DATASET_PATH)
    X_meta, y_meta = [], []

    for path, label in tqdm(zip(image_paths, labels), total=len(labels)):
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                dl_probs = torch.softmax(model(img_tensor), dim=1).cpu().numpy().flatten()

            glcm_feat = extract_glcm_features(path)
            knn_probs = knn.predict_proba(glcm_feat).flatten()

            stacked_input = np.concatenate((dl_probs, knn_probs))
            X_meta.append(stacked_input)
            y_meta.append(label)
        except Exception as e:
            print(f"[WARN] Skipped {path}: {e}")

    print("[INFO] Training logistic regression meta-learner...")
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(X_meta, y_meta)

    preds = meta_clf.predict(X_meta)
    report = classification_report(y_meta, preds, target_names=CLASS_NAMES)
    print(report)

    print(f"[INFO] Saving meta-learner to {OUTPUT_META_MODEL_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_META_MODEL_PATH), exist_ok=True)
    joblib.dump(meta_clf, OUTPUT_META_MODEL_PATH)
    print("[✅] Meta-learner training complete.")
