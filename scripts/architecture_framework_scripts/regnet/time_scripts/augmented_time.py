import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets # datasets needed for ImageFolder
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
import time # Added for timing

# --- Config ---
ARCH_NAME = "regnet"
INFERENCE_TYPE_TAG = "augmented" # Defines the 'inference_type' part of the path

NUM_CLASSES = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [ # Ensure order matches training and dataset folder names
    "Bacterial Spot", "Cercospora Leaf Spot", "Common Rust", "Early Blight",
    "Healthy", "Late Blight", "Northern Leaf Blight",
    "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]
KNN_MODEL_PATH = "models/glcm_knn/trained/glcm_knn.pkl"
MODEL_PATH = "models/regnet/augmented/regnet_augmented.pth" # Path to the AugmentedRegNet model state_dict

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- GLCM Feature Extractor ---
def extract_glcm_features(image_path): # Robust version
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
        # print(f"[WARN] Image {image_path} appears blank/uniform. GLCM returning zeros.")
        return np.zeros((1,6)) # 6 GLCM features

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ]).reshape(1,-1) # Ensure it's (1, 6)

# --- Augmented Model ---
class AugmentedRegNet(nn.Module):
    def __init__(self, num_input_knn_features=NUM_CLASSES): # KNN probas will have NUM_CLASSES features
        super().__init__()
        # Load RegNet backbone, use default weights if you trained on top of it,
        # or None if your MODEL_PATH contains the full model including backbone weights.
        # Original script used DEFAULT, implies fine-tuning or using features.
        # For full model load, often use weights=None and load complete state_dict.
        # Let's assume weights=None, as the full model state_dict is loaded later.
        self.backbone = models.regnet_y_800mf(weights=None) # Or models.RegNet_Y_800MF_Weights.DEFAULT
        self.backbone.fc = nn.Identity() # Remove original FC layer to get features

        # Dynamically determine feature dimension from backbone
        with torch.no_grad():
            dummy_input_for_dim_check = torch.randn(1, 3, 224, 224)
            feature_dim = self.backbone(dummy_input_for_dim_check).shape[1]
        
        self.feature_dim = feature_dim
        self.num_input_knn_features = num_input_knn_features

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + self.num_input_knn_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, img_tensor, knn_feature_tensor): # Renamed knn_tensor for clarity
        image_features = self.backbone(img_tensor)
        combined_features = torch.cat((image_features, knn_feature_tensor), dim=1)
        return self.classifier(combined_features)

    def load_and_prepare_model(self, model_state_dict_path, device_to_use):
        print(f"[INFO] Loading AugmentedRegNet state_dict from: {model_state_dict_path}")
        try:
            self.load_state_dict(torch.load(model_state_dict_path, map_location=device_to_use))
        except FileNotFoundError:
            print(f"[ERROR] Model state_dict not found at: {model_state_dict_path}")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to load AugmentedRegNet state_dict: {e}")
            raise

        self.to(device_to_use).eval()

        # Warm-up for the entire AugmentedRegNet model
        if device_to_use.type == 'cuda':
            print("[INFO] Performing GPU warm-up for AugmentedRegNet model...")
            try:
                dummy_img_tensor = torch.randn(1, 3, 224, 224).to(device_to_use)
                # KNN features/probas tensor, shape (1, num_input_knn_features)
                dummy_knn_tensor = torch.randn(1, self.num_input_knn_features).to(device_to_use)
                with torch.no_grad():
                    for _ in range(5): # Multiple warm-up inferences
                        _ = self.forward(dummy_img_tensor, dummy_knn_tensor)
                torch.cuda.synchronize()
                print("[INFO] AugmentedRegNet GPU warm-up complete.")
            except Exception as e:
                print(f"[WARN] Error during AugmentedRegNet warm-up: {e}")
        return self

# --- Dataset iteration (using ImageFolder.samples directly) ---
# No separate get_image_paths_and_labels needed if using dataset.samples

# --- Save Classification Metrics and Confusion Matrix (per dataset_type) ---
def save_per_dataset_classification_results(report, conf_mat, arch_name_conf, inf_type_tag_conf, ds_type_name):
    # Reusing standardized function
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
    # Reusing standardized function
    output_dir_combined = os.path.join("results", arch_name_conf, inf_type_tag_conf)
    os.makedirs(output_dir_combined, exist_ok=True)
    df_timing = pd.DataFrame(all_datasets_timing_list)
    df_timing.to_csv(os.path.join(output_dir_combined, "inference_times.csv"), index=False)
    with open(os.path.join(output_dir_combined, "inference_times.txt"), 'w') as f:
        f.write(f"Combined Inference Timing Summary for {arch_name_conf} - {inf_type_tag_conf}:\n\n{df_timing.to_string(index=False)}")
    print(f"[INFO] Combined inference timing results saved to: {output_dir_combined}")

# --- Inference for a single dataset ---
def run_augmented_inference_on_dataset(dataset_root_path, dataset_name, arch_name_config, inference_type_tag_config):
    print(f"\n[INFO] Running augmented inference (WITH WARM-UP) on {dataset_name} (Arch: {arch_name_config}, Type: {inference_type_tag_config})...")

    try:
        # Using datasets.ImageFolder to get image paths and true labels
        image_dataset = datasets.ImageFolder(dataset_root_path)
        if not image_dataset.samples:
            raise ValueError(f"No image samples found in {dataset_root_path}")
        print(f"[INFO] Loaded {len(image_dataset.samples)} samples from {dataset_name}.")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset '{dataset_name}' from {dataset_root_path}: {e}. Skipping.")
        return None

    # Load AugmentedRegNet model (includes warm-up) and KNN model
    # num_input_knn_features should match the output of knn.predict_proba, which is NUM_CLASSES
    augmented_model = AugmentedRegNet(num_input_knn_features=NUM_CLASSES)
    augmented_model.load_and_prepare_model(MODEL_PATH, DEVICE)

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

    print(f"[INFO] Starting inference loop for {len(image_dataset.samples)} images in {dataset_name}...")
    # dataset.samples is a list of (image_path, class_index)
    for img_path, current_true_label in tqdm(image_dataset.samples, total=len(image_dataset.samples), desc=f"Inferring on {dataset_name}"):
        try:
            # Optional: sync before timing each image
            # if DEVICE.type == 'cuda': torch.cuda.synchronize()
            start_image_time = time.perf_counter()

            # 1. GLCM feature extraction
            glcm_feat_np = extract_glcm_features(img_path) # Shape (1, 6)

            # 2. KNN probability prediction
            # knn.predict_proba returns shape (1, num_classes)
            knn_proba_np = knn_glcm_model.predict_proba(glcm_feat_np).flatten() # Flatten to 1D array
            if len(knn_proba_np) != NUM_CLASSES:
                raise ValueError(f"KNN proba vector length mismatch. Expected {NUM_CLASSES}, Got {len(knn_proba_np)} for {img_path}")

            # 3. Image transformation for DL model
            pil_image = Image.open(img_path).convert("RGB")
            img_tensor_transformed = TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

            # 4. Convert KNN probabilities to tensor
            knn_proba_tensor = torch.tensor(knn_proba_np, dtype=torch.float32).unsqueeze(0).to(DEVICE) # Shape (1, NUM_CLASSES)

            # 5. Forward pass through AugmentedRegNet
            with torch.no_grad():
                # Sync before model call if there were preceding CUDA ops on this stream (unlikely here for KNN part)
                # if DEVICE.type == 'cuda': torch.cuda.synchronize()
                output_logits = augmented_model(img_tensor_transformed, knn_proba_tensor)
                # Sync after model call before reading results or stopping timer
                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                final_predicted_class = torch.argmax(output_logits, dim=1).item()

            end_image_time = time.perf_counter()

            # Record results and time if successful
            total_inference_time_seconds += (end_image_time - start_image_time)
            predicted_labels_for_report.append(final_predicted_class)
            true_labels_for_report.append(current_true_label)
            processed_image_count += 1

        except ValueError as ve:
            print(f"[WARN] Skipping image {img_path} due to ValueError: {ve}")
        except Exception as e:
            print(f"[WARN] An unexpected error occurred processing {img_path}. Skipped. Error: {e}")
            # import traceback; traceback.print_exc() # For detailed debugging

    if processed_image_count == 0:
        print(f"[ERROR] No images were successfully processed for {dataset_name}. Cannot generate report or timings.")
        return None

    avg_time_per_image = total_inference_time_seconds / processed_image_count if processed_image_count > 0 else 0

    print(f"\n[INFO] --- TIMING RESULTS for {dataset_name} ({arch_name_config} - {inference_type_tag_config}) ---")
    print(f"[INFO] Total images processed: {processed_image_count} / {len(image_dataset.samples)}")
    print(f"[INFO] Total inference time (WITH WARM-UP): {total_inference_time_seconds:.4f} seconds")
    print(f"[INFO] Average time per image (WITH WARM-UP): {avg_time_per_image:.6f} seconds")

    report_dict = classification_report(true_labels_for_report, predicted_labels_for_report, target_names=CLASS_NAMES, labels=np.arange(NUM_CLASSES), output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(true_labels_for_report, predicted_labels_for_report, labels=np.arange(NUM_CLASSES))

    save_per_dataset_classification_results(report_dict, conf_mat, arch_name_config, inference_type_tag_config, dataset_name)

    timing_data = {
        "Inference Type": dataset_name, # Dataset name (e.g., plantvillage)
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
    print("[INFO] NOTE: Inference timing will be performed WITH explicit model warm-up.")

    datasets_to_process = {
        "plantvillage": "dataset/plantvillage/test",
        "plantdoc": "dataset/plantdoc/test"
    }
    all_timing_results_summary = []

    for dataset_name_key, dataset_actual_path in datasets_to_process.items():
        if not os.path.exists(dataset_actual_path) or not os.path.isdir(dataset_actual_path):
            print(f"[WARN] Dataset path not found or invalid: {dataset_actual_path} for '{dataset_name_key}'. Skipping.")
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