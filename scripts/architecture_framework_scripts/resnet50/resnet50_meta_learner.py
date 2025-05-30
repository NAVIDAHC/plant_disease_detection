import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import joblib
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd

# --- Config ---
DATASET_PATH = "dataset/plantdoc/val"  # use validation set for stacking
KNN_MODEL_PATH = "models/glcm_knn/trained/glcm_knn.pkl"
DL_MODEL_PATH = "models/resnet50/trained/resnet50_best_model.pth"
OUTPUT_META_MODEL_PATH = "models/resnet50/meta_learner/resnet50_meta_learner.pkl"
NUM_CLASSES = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "Bacterial Spot", "Cercospora Leaf Spot", "Common Rust", "Early Blight",
    "Healthy", "Late Blight", "Northern Leaf Blight",
    "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- GLCM Extractor ---
def extract_glcm_features(image_path):
    import cv2
    from skimage.feature import graycomatrix, graycoprops
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0],
    ]
    return np.array(features).reshape(1, -1)

# --- DL Model Loader ---
def load_resnet50_model(path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# --- Dataset Loader ---
def get_image_paths_and_labels(root_dir):
    image_paths, labels = [], []
    class_to_idx = {cls: i for i, cls in enumerate(sorted(os.listdir(root_dir)))}
    for cls in os.listdir(root_dir):
        cls_folder = os.path.join(root_dir, cls)
        for fname in os.listdir(cls_folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(cls_folder, fname))
                labels.append(class_to_idx[cls])
    return image_paths, labels

# --- Main ---
if __name__ == "__main__":
    print("[INFO] Loading models...")
    knn = joblib.load(KNN_MODEL_PATH)
    model = load_resnet50_model(DL_MODEL_PATH)

    print("[INFO] Loading validation images for stacking...")
    image_paths, labels = get_image_paths_and_labels(DATASET_PATH)
    X_meta, y_meta = [], []

    for img_path, label in tqdm(zip(image_paths, labels), total=len(labels)):
        try:
            # Deep model prediction
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(img_tensor)
                dl_probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

            # KNN prediction
            glcm_feat = extract_glcm_features(img_path)
            knn_probs = knn.predict_proba(glcm_feat).flatten()

            # Combine
            stacked_input = np.concatenate((dl_probs, knn_probs))
            X_meta.append(stacked_input)
            y_meta.append(label)

        except Exception as e:
            print(f"[WARN] Skipping {img_path}: {e}")

    print("[INFO] Training meta-classifier (Logistic Regression)...")
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(X_meta, y_meta)

    preds = meta_clf.predict(X_meta)
    report = classification_report(y_meta, preds, target_names=CLASS_NAMES)
    print(report)

    print(f"[INFO] Saving meta-classifier to {OUTPUT_META_MODEL_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_META_MODEL_PATH), exist_ok=True)
    joblib.dump(meta_clf, OUTPUT_META_MODEL_PATH)
    print("[✅] Meta-classifier saved.")
