import os
import cv2
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time # Import the time module

# --- Config ---
ARCH_NAME = "glcm_knn"
NUM_CLASSES = 9
KNN_MODEL_PATH = "models/glcm_knn/trained/glcm_knn.pkl"
CLASS_NAMES = [
    "Bacterial Spot", "Cercospora Leaf Spot", "Common Rust", "Early Blight",
    "Healthy", "Late Blight", "Northern Leaf Blight",
    "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]

# --- GLCM Feature Extractor ---
def extract_glcm_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Ensure image is 8-bit for graycomatrix
    if gray.dtype != np.uint8:
        gray = (gray / gray.max() * 255).astype(np.uint8)

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ]

# --- Image Path Loader ---
def get_image_paths_and_labels(root_dir):
    image_paths, labels = [], []
    class_to_idx = {cls: i for i, cls in enumerate(sorted(os.listdir(root_dir)))}
    for cls in sorted(os.listdir(root_dir)):
        cls_dir = os.path.join(root_dir, cls)
        if os.path.isdir(cls_dir): # Check if it's a directory
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(cls_dir, fname))
                    labels.append(class_to_idx[cls])
    return image_paths, labels

# --- Inference ---
def run_glcm_knn_inference(dataset_path, dataset_name):
    print(f"[INFO] Running GLCM+KNN inference on {dataset_name}...")
    image_paths, labels = get_image_paths_and_labels(dataset_path)
    knn = joblib.load(KNN_MODEL_PATH)

    y_true, y_pred = [], []
    start_time = time.time() # Record start time

    for path, label in tqdm(zip(image_paths, labels), total=len(labels)):
        try:
            glcm_feat = extract_glcm_features(path)
            pred = knn.predict([glcm_feat])[0]
            y_true.append(label)
            y_pred.append(pred)
        except Exception as e:
            print(f"[WARN] Skipped {path}: {e}")

    end_time = time.time() # Record end time
    total_time = end_time - start_time
    num_images = len(y_true) # Use the number of successfully processed images
    avg_time_per_image = total_time / num_images if num_images > 0 else 0 # Avoid division by zero

    # Save inference time
    results_dir = "results/glcm_knn"
    os.makedirs(results_dir, exist_ok=True)

    inference_time_data = {
        "Inference Type": [dataset_name],
        "Total Time (s)": [total_time],
        "Avg Time per Image (s)": [avg_time_per_image]
    }
    inference_time_df = pd.DataFrame(inference_time_data)

    # Save to CSV
    csv_path = os.path.join(results_dir, "inference_time.csv")
    if os.path.exists(csv_path):
        inference_time_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        inference_time_df.to_csv(csv_path, index=False)


    # Save to TXT
    txt_path = os.path.join(results_dir, "inference_time.txt")
    with open(txt_path, "a") as f:
        # Add header only if file is empty
        if os.stat(txt_path).st_size == 0:
             f.write("Inference Type,Total Time (s),Avg Time per Image (s)\n")
        f.write(f"{dataset_name},{total_time},{avg_time_per_image}\n")


    out_dir = f"results/{ARCH_NAME}/base_model/{dataset_name}"
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
    plt.title(f"{ARCH_NAME.replace('_', ' ').title()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{ARCH_NAME}_confusion_matrix.png"))
    plt.close()

    print(f"[✅] Results saved to: {out_dir}")
    print(f"[✅] Inference time for {dataset_name} saved to: {results_dir}/inference_time.csv and {results_dir}/inference_time.txt")


# --- Main ---
if __name__ == "__main__":
    # Ensure the main results directory exists
    os.makedirs("results/glcm_knn", exist_ok=True)
    # Clear the inference_time files if they exist to start fresh
    for file_ext in ['.csv', '.txt']:
        file_path = os.path.join("results/glcm_knn", f"inference_time{file_ext}")
        if os.path.exists(file_path):
            os.remove(file_path)

    run_glcm_knn_inference("dataset/plantvillage/test", "PlantVillage")
    run_glcm_knn_inference("dataset/plantdoc/test", "PlantDoc")