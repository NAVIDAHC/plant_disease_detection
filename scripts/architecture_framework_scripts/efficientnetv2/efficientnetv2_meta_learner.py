import os
import joblib
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import cv2
from skimage.feature import graycomatrix, graycoprops

# --- Config ---
DATA_DIR = "dataset/plantdoc/val"
KNN_MODEL_PATH = "models/glcm_knn/trained/glcm_knn.pkl"
DL_MODEL_PATH = "models/efficientnetv2/trained/efficientnetv2_best_model.pth"
META_MODEL_PATH = "models/efficientnetv2/meta_learner/efficientnet_meta_learner.pkl"
NUM_CLASSES = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- EfficientNet preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- GLCM feature extractor (matches your new training script) ---
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
    return np.array(features).reshape(1, -1)

# --- Dataset loader ---
def get_image_paths_and_labels(data_dir):
    class_names = sorted(os.listdir(data_dir))
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    paths, labels = [], []
    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_path, fname))
                labels.append(class_to_idx[cls])
    return paths, labels

# --- Model loaders ---
def load_knn(path):
    return joblib.load(path)

def load_efficientnet(path):
    model = efficientnet_v2_s()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()

# --- Main ---
if __name__ == "__main__":
    print("[INFO] Loading models...")
    knn = load_knn(KNN_MODEL_PATH)
    effnet = load_efficientnet(DL_MODEL_PATH)

    print("[INFO] Loading dataset...")
    image_paths, labels = get_image_paths_and_labels(DATA_DIR)
    X_meta, y_meta = [], []

    print("[INFO] Extracting predictions...")
    for img_path, label in tqdm(zip(image_paths, labels), total=len(labels)):
        # EfficientNet prediction
        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                eff_logits = effnet(img_tensor)
                eff_probs = F.softmax(eff_logits, dim=1).cpu().numpy().flatten()
        except Exception as e:
            print(f"[EffNet error] {img_path}: {e}")
            continue

        # GLCM + KNN prediction
        try:
            glcm_feats = extract_glcm_features(img_path)
            knn_probs = knn.predict_proba(glcm_feats).flatten()
        except Exception as e:
            print(f"[GLCM error] {img_path}: {e}")
            continue

        # Combine
        stacked_input = np.concatenate((eff_probs, knn_probs))
        X_meta.append(stacked_input)
        y_meta.append(label)

    # Train meta-learner
    print("[INFO] Training logistic regression...")
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(X_meta, y_meta)

    print("[INFO] Evaluation on same set:")
    preds = meta_clf.predict(X_meta)
    print(classification_report(y_meta, preds))

    print(f"[INFO] Saving meta-classifier to {META_MODEL_PATH}")
    os.makedirs(os.path.dirname(META_MODEL_PATH), exist_ok=True)
    joblib.dump(meta_clf, META_MODEL_PATH)

    print("[✅ DONE] Stacking meta-classifier saved.")
