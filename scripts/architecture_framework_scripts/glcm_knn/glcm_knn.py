import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import graycomatrix, graycoprops

# --- Config ---
DATA_DIR = "dataset/PlantVillage_modified/Train"
MODEL_SAVE_PATH = "scripts/architecture_framework_scripts/combination/glcm+knn_for_combination.pkl"
N_NEIGHBORS = 3  # You can tune this

# --- Feature extraction ---
def extract_glcm_features(image_path):
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
    return np.array(features)

# --- Load data and extract features ---
def load_dataset(data_dir):
    X, y = [], []
    class_names = sorted(os.listdir(data_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    img_path = os.path.join(cls_path, fname)
                    features = extract_glcm_features(img_path)
                    X.append(features)
                    y.append(class_to_idx[cls])
                except Exception as e:
                    print(f"⚠️ Failed on {fname}: {e}")

    return np.array(X), np.array(y), class_names

# --- Main ---
if __name__ == "__main__":
    print("[INFO] Loading dataset...")
    X, y, class_names = load_dataset(DATA_DIR)

    print(f"[INFO] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")

    print("[INFO] Training KNN model...")
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    knn.fit(X, y)

    print("[INFO] Evaluating on training set...")
    y_pred = knn.predict(X)
    print(classification_report(y, y_pred, target_names=class_names))

    print("[INFO] Saving KNN model...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(knn, MODEL_SAVE_PATH)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("GLCM + KNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig("glcm_knn_confusion_matrix.png")
    print("[INFO] Training complete. Model and confusion matrix saved.")



KNN_PATH = "scripts/architecture_framework_scripts/combination/glcm+knn_for_combination.pkl"
EFFNET_PATH = "models/architectures_data augmented/efficientnet/efficientnet.pth"
META_PATH = "models/meta_classifier.pkl"