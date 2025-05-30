import os
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from skimage.feature import graycomatrix, graycoprops
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Config ---
NUM_CLASSES = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- File paths ---
KNN_PATH = "scripts/architecture_framework_scripts/combination/glcm+knn_for_combination.pkl"
EFFNET_PATH = "models/architectures_cross_validated/cross_validated_models/efficientnet_cross_validated/efficientnetv2_plantvillage_best_fold1.pth"
META_PATH = "models/meta_classifier1.pkl"
VAL_FOLDER = "dataset/PlantDoc/Test"  # 👈 Your fixed folder path
# OUTPUT_DIR = 

# --- Class names ---
CLASS_NAMES = [
    "Bacterial Spot", "Cercospora Leaf Spot", "Common Rust", "Early Blight",
    "Healthy", "Late Blight", "Northern Leaf Blight",
    "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Loaders ---
def load_effnet(path):
    model = efficientnet_v2_s()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def extract_glcm_features(path):
    img = cv2.imread(path)
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

def predict_image(img_path, effnet, knn, meta):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        effnet_out = effnet(img_tensor)
        effnet_probs = F.softmax(effnet_out, dim=1).cpu().numpy().flatten()

    glcm_feat = extract_glcm_features(img_path)
    knn_probs = knn.predict_proba(glcm_feat).flatten()

    stacked = np.concatenate((effnet_probs, knn_probs)).reshape(1, -1)
    pred = meta.predict(stacked)[0]
    return pred

# --- Evaluation ---
def run_inference_on_folder(folder_path, effnet, knn, meta):
    image_paths = []
    true_labels = []
    pred_labels = []

    for class_name in os.listdir(folder_path):
        class_dir = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_dir): continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")): continue
            img_path = os.path.join(class_dir, fname)
            try:
                pred = predict_image(img_path, effnet, knn, meta)
                image_paths.append(fname)
                true_labels.append(CLASS_TO_IDX[class_name])
                pred_labels.append(pred)
            except Exception as e:
                print(f"[⚠️] Error on {img_path}: {e}")

    return image_paths, true_labels, pred_labels

# --- Save Outputs ---
def save_outputs(imgs, y_true, y_pred):
    output_dir = "results/efficientnet_plantdoc"
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV predictions
    df = pd.DataFrame({
        "Image": imgs,
        "True_Label": [CLASS_NAMES[i] for i in y_true],
        "Predicted_Label": [CLASS_NAMES[i] for i in y_pred]
    })
    df.to_csv(os.path.join(output_dir, "stacking_predictions.csv"), index=False)

    # Save classification report (TXT)
    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    with open(os.path.join(output_dir, "stacking_classification_report.txt"), "w") as f:
        f.write(report_text)

    # Save classification report (CSV)
    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(output_dir, "stacking_classification_report.csv"), index=True)

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Stacking Inference - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stacking_confusion_matrix.png"))
    plt.close()

    print(f"[✅] Results saved to: {output_dir}")

# --- Main ---
if __name__ == "__main__":
    print("[INFO] Loading models...")
    effnet = load_effnet(EFFNET_PATH)
    knn = joblib.load(KNN_PATH)
    meta = joblib.load(META_PATH)

    print(f"[INFO] Running inference on folder: {VAL_FOLDER}")
    imgs, y_true, y_pred = run_inference_on_folder(VAL_FOLDER, effnet, knn, meta)

    print("[INFO] Saving results...")
    save_outputs(imgs, y_true, y_pred)
