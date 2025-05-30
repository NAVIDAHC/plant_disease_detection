import os
import torch
import torch.nn as nn
from torchvision import models, transforms # Removed datasets as it's not used directly here
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
import time # Added for timing

# --- Config ---
ARCH_NAME = "resnet50"
INFERENCE_TYPE_TAG = "augmented" # Defines the 'inference_type' part of the path

NUM_CLASSES = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [ # Ensure this order matches your model's training and dataset folder names
    "Bacterial Spot", "Cercospora Leaf Spot", "Common Rust", "Early Blight",
    "Healthy", "Late Blight", "Northern Leaf Blight",
    "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]
DL_MODEL_PATH = "models/resnet50/augmented/resnet50_augmented_model.pth" # Path to the AugmentedResNet50 model
KNN_MODEL_PATH = "models/glcm_knn/trained/glcm_knn.pkl"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Model ---
class AugmentedResNet50(nn.Module):
    def __init__(self, num_glcm_features=NUM_CLASSES): # Make num_glcm_features (from KNN probas) configurable
        super().__init__()
        self.backbone = models.resnet50(weights=None) # Load without pretrained weights, expecting full state_dict
        self.backbone.fc = nn.Identity() # Remove final layer to get features
        self.feature_dim = 2048 # ResNet50 feature dimension
        
        # Classifier takes concatenated ResNet features and KNN probability vector
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + num_glcm_features, 256), # Input size = ResNet feats + KNN probas
            nn.ReLU(),
            nn.Dropout(0.3), # Matches original dropout
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, img_tensor, knn_proba_vector): # Changed knn_vector to knn_proba_vector for clarity
        image_features = self.backbone(img_tensor)
        # Ensure dimensions are compatible for concatenation (batch dimension)
        combined_features = torch.cat((image_features, knn_proba_vector), dim=1)
        return self.classifier(combined_features)

    def load_and_prepare_model(self, model_path, device_to_use):
        self.load_state_dict(torch.load(model_path, map_location=device_to_use))
        self.to(device_to_use).eval()
        # Warm-up
        if device_to_use.type == 'cuda':
            print("[INFO] Performing GPU warm-up for AugmentedResNet50 model...")
            try:
                dummy_img_tensor = torch.randn(1, 3, 224, 224).to(device_to_use)
                # KNN probas will have shape (1, NUM_CLASSES)
                dummy_knn_tensor = torch.randn(1, NUM_CLASSES).to(device_to_use)
                with torch.no_grad():
                    for _ in range(5): # Multiple warm-up inferences
                        _ = self.forward(dummy_img_tensor, dummy_knn_tensor)
                torch.cuda.synchronize()
                print("[INFO] AugmentedResNet50 GPU warm-up complete.")
            except Exception as e:
                print(f"[WARN] Error during AugmentedResNet50 warm-up: {e}")
        return self

# --- GLCM ---
def extract_glcm_features(image_path): # Same robust version as before
    img = cv2.imread(image_path)
    if img is None:
        try:
            pil_img = Image.open(image_path).convert('L')
            gray = np.array(pil_img)
        except Exception as e_pil:
            raise ValueError(f"Image could not be loaded by OpenCV or PIL: {image_path}. PIL error: {e_pil}")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if gray.ndim != 2:
        raise ValueError(f"Grayscale image is not 2D. Path: {image_path}, Shape: {gray.shape}")
    if np.all(gray == gray[0,0]):
         # print(f"[WARN] Image {image_path} appears to be blank or uniform. GLCM might be non-informative, returning zeros.")
         return np.zeros((1,6)) # 6 GLCM features

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
def get_image_paths_and_labels(root_dir, class_names_config): # Same robust version
    image_paths, labels = [], []
    class_subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if not class_subdirs:
        raise ValueError(f"No subdirectories found in {root_dir}. Check dataset structure.")

    try:
        class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names_config)}
    except TypeError:
        print("[ERROR] CLASS_NAMES is not defined correctly as a list.")
        raise

    actual_found_classes_count = 0
    for class_dir_name in class_subdirs:
        if class_dir_name not in class_names_config:
            # print(f"[WARN] Directory '{class_dir_name}' in dataset not in CLASS_NAMES. Skipping.")
            continue
        actual_found_classes_count +=1
        cls_idx = class_to_idx[class_dir_name]
        cls_path = os.path.join(root_dir, class_dir_name)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(cls_path, fname))
                labels.append(cls_idx)
    
    if not image_paths:
        raise ValueError(f"No images found in {root_dir} for configured classes. Check dataset structure and CLASS_NAMES.")
    
    print(f"[INFO] Found {len(image_paths)} images in {actual_found_classes_count} recognized classes from {root_dir}.")
    return image_paths, labels

# --- Save Classification Metrics and Confusion Matrix (per dataset_type) ---
def save_per_dataset_classification_results(report, conf_mat, arch_name_conf, inf_type_tag_conf, ds_type_name):
    output_dir = os.path.join("results", arch_name_conf, inf_type_tag_conf, ds_type_name)
    os.makedirs(output_dir, exist_ok=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f"{arch_name_conf}_{inf_type_tag_conf}_{ds_type_name}_classification_report.csv"))
    with open(os.path.join(output_dir, f"{arch_name_conf}_{inf_type_tag_conf}_{ds_type_name}_classification_report.txt"), 'w') as f:
        f.write(f"Classification Report for {arch_name_conf} ({inf_type_tag_conf}) on {ds_type_name}:\n\n{report_df.to_string()}")
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 8})
    plt.title(f"Confusion Matrix - {arch_name_conf} ({inf_type_tag_conf}) on {ds_type_name}", fontsize=10)
    plt.xlabel("Predicted", fontsize=9); plt.ylabel("Actual", fontsize=9)
    plt.xticks(fontsize=8, rotation=45, ha="right"); plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{arch_name_conf}_{inf_type_tag_conf}_{ds_type_name}_confusion_matrix.png"))
    plt.close()
    print(f"[INFO] Per-dataset classification results saved to: {output_dir}")

# --- Save Combined Inference Timing Results ---
def save_combined_inference_times(all_datasets_timing_list, arch_name_conf, inf_type_tag_conf):
    output_dir_combined = os.path.join("results", arch_name_conf, inf_type_tag_conf)
    os.makedirs(output_dir_combined, exist_ok=True)
    df_timing = pd.DataFrame(all_datasets_timing_list)
    df_timing.to_csv(os.path.join(output_dir_combined, "inference_times.csv"), index=False)
    with open(os.path.join(output_dir_combined, "inference_times.txt"), 'w') as f:
        f.write(f"Combined Inference Timing Summary for {arch_name_conf} - {inf_type_tag_conf}:\n\n{df_timing.to_string(index=False)}")
    print(f"[INFO] Combined inference timing results saved to: {output_dir_combined}")

# --- Inference for a single dataset ---
def run_augmented_inference_on_dataset(dataset_path, dataset_name, arch_name_config, inference_type_tag_config):
    print(f"\n[INFO] Running augmented inference on {dataset_name} (Arch: {arch_name_config}, Type: {inference_type_tag_config})...")
    
    try:
        image_paths, true_labels_all = get_image_paths_and_labels(dataset_path, CLASS_NAMES)
    except ValueError as e:
        print(f"[ERROR] Could not load images/labels for {dataset_name}: {e}. Skipping.")
        return None

    # Load AugmentedResNet50 model and KNN model
    augmented_model = AugmentedResNet50(num_glcm_features=NUM_CLASSES) # KNN predict_proba gives NUM_CLASSES probabilities
    augmented_model.load_and_prepare_model(DL_MODEL_PATH, DEVICE) # This also does warm-up

    try:
        knn_glcm_model = joblib.load(KNN_MODEL_PATH)
    except FileNotFoundError:
        print(f"[ERROR] KNN model not found at {KNN_MODEL_PATH}. Cannot proceed.")
        return None
    except Exception as e:
        print(f"[ERROR] Could not load KNN model: {e}")
        return None

    true_labels_for_report = []
    predicted_labels_for_report = []
    total_inference_time_seconds = 0.0
    processed_image_count = 0

    print(f"[INFO] Starting inference loop for {len(image_paths)} images in {dataset_name}...")
    for img_path, current_true_label in tqdm(zip(image_paths, true_labels_all), total=len(true_labels_all), desc=f"Inferring on {dataset_name}"):
        try:
            start_image_time = time.perf_counter() # Time the entire sequence per image

            # 1. GLCM feature extraction
            glcm_feat_np = extract_glcm_features(img_path)
            
            # 2. KNN probability prediction
            # knn.predict_proba returns shape (1, num_classes)
            knn_proba_np = knn_glcm_model.predict_proba(glcm_feat_np).flatten() # Flatten to 1D array of NUM_CLASSES length
            
            # 3. Image transformation for DL model
            pil_image = Image.open(img_path).convert("RGB")
            img_tensor_transformed = TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
            
            # 4. Convert KNN probabilities to tensor
            knn_proba_tensor = torch.tensor(knn_proba_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            if knn_proba_tensor.shape[1] != NUM_CLASSES: # Sanity check
                 raise ValueError(f"KNN proba tensor shape mismatch. Expected ({1},{NUM_CLASSES}), Got {knn_proba_tensor.shape}")


            # 5. Forward pass through AugmentedResNet50
            with torch.no_grad():
                output_logits = augmented_model(img_tensor_transformed, knn_proba_tensor)
                final_predicted_class = torch.argmax(output_logits, dim=1).item()
            
            end_image_time = time.perf_counter()

            # Record results and time if successful
            total_inference_time_seconds += (end_image_time - start_image_time)
            predicted_labels_for_report.append(final_predicted_class)
            true_labels_for_report.append(current_true_label)
            processed_image_count += 1

        except ValueError as ve: # Catch specific errors from GLCM or shape mismatches
            print(f"[WARN] Skipping image {img_path} due to ValueError: {ve}")
        except Exception as e:
            print(f"[WARN] An unexpected error occurred processing {img_path}. Skipped. Error: {e}")

    if processed_image_count == 0:
        print(f"[ERROR] No images were successfully processed for {dataset_name}. Cannot generate report or timings.")
        return None

    avg_time_per_image = total_inference_time_seconds / processed_image_count if processed_image_count > 0 else 0

    print(f"\n[INFO] --- TIMING RESULTS for {dataset_name} ({arch_name_config} - {inference_type_tag_config}) ---")
    print(f"[INFO] Total images processed: {processed_image_count} / {len(image_paths)}")
    print(f"[INFO] Total inference time: {total_inference_time_seconds:.4f} seconds")
    print(f"[INFO] Average time per image: {avg_time_per_image:.6f} seconds")

    report_dict = classification_report(true_labels_for_report, predicted_labels_for_report, target_names=CLASS_NAMES, labels=np.arange(NUM_CLASSES), output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(true_labels_for_report, predicted_labels_for_report, labels=np.arange(NUM_CLASSES))

    save_per_dataset_classification_results(report_dict, conf_mat, arch_name_config, inference_type_tag_config, dataset_name)

    timing_data = {
        "Inference Type": dataset_name, # This refers to the dataset (e.g. plantvillage)
        "Total Time (s)": total_inference_time_seconds,
        "Avg Time per Image (s)": avg_time_per_image
    }
    return timing_data

# --- Main Execution ---
if __name__ == "__main__":
    print(f"[INFO] Script Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Architecture: {ARCH_NAME}")
    print(f"[INFO] Inference Type Tag: {INFERENCE_TYPE_TAG}")

    datasets_to_process = {
        "plantvillage": "dataset/plantvillage/test",
        "plantdoc": "dataset/plantdoc/test"
    }
    all_timing_results_summary = []

    for dataset_name_key, dataset_actual_path in datasets_to_process.items():
        if not os.path.exists(dataset_actual_path):
            print(f"[WARN] Dataset path not found: {dataset_actual_path} for '{dataset_name_key}'. Skipping.")
            continue
        
        timing_result = run_augmented_inference_on_dataset(
            dataset_actual_path,
            dataset_name_key,
            ARCH_NAME,
            INFERENCE_TYPE_TAG
        )
        if timing_result:
            all_timing_results_summary.append(timing_result)

    if all_timing_results_summary:
        save_combined_inference_times(all_timing_results_summary, ARCH_NAME, INFERENCE_TYPE_TAG)
    else:
        print("[WARN] No timing results were generated from any dataset to save in the combined file.")

    print(f"\n[INFO] Script End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("[INFO] All operations complete.")