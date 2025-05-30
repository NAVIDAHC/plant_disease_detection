import os
import torch
import torch.nn as nn
from torchvision import models, transforms
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
ARCH_NAME = "resnet50"
NUM_CLASSES = 9
HEALTHY_CLASS_INDEX = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "Bacterial Spot", "Cercospora Leaf Spot", "Common Rust", "Early Blight",
    "Healthy", "Late Blight", "Northern Leaf Blight",
    "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]
DL_MODEL_PATH = "models/resnet50/trained/resnet50_best_model.pth"
KNN_MODEL_PATH = "models/glcm_knn/trained/glcm_knn.pkl"
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load ResNet50 ---
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(DL_MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()

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
    ]).reshape(1, -1)

# --- Image Loader ---
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

# --- Inference ---
def run_hierarchical_inference(dataset_path, dataset_name):
    print(f"[INFO] Running hierarchical inference on {dataset_name}...")
    image_paths, labels = get_image_paths_and_labels(dataset_path)
    model = load_model()
    knn = joblib.load(KNN_MODEL_PATH)

    y_true, y_pred = [], []

    for path, label in tqdm(zip(image_paths, labels), total=len(labels)):
        try:
            glcm_feat = extract_glcm_features(path)
            knn_pred = knn.predict(glcm_feat)[0]

            if knn_pred == HEALTHY_CLASS_INDEX:
                final_pred = knn_pred
            else:
                img = Image.open(path).convert("RGB")
                input_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    final_pred = torch.argmax(model(input_tensor), dim=1).item()

            y_true.append(label)
            y_pred.append(final_pred)
        except Exception as e:
            print(f"[WARN] Skipped {path}: {e}")

    out_dir = f"results/{ARCH_NAME}/hierarchical/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)

    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(out_dir, f"{ARCH_NAME}_inference_result.csv"), index=True)
    with open(os.path.join(out_dir, f"{ARCH_NAME}_inference_result.txt"), "w") as f:
        f.write(report_df.to_string())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{ARCH_NAME.replace('_', ' ').title()} Hierarchical Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{ARCH_NAME}_confusion_matrix.png"))
    plt.close()

    print(f"[✅] Results saved to: {out_dir}")

# --- Main ---
if __name__ == "__main__":
    run_hierarchical_inference("dataset/plantvillage/test", "plantvillage")
    run_hierarchical_inference("dataset/plantdoc/test", "plantdoc")
