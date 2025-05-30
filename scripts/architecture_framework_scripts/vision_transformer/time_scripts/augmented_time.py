import os
import torch
import torch.nn as nn
# import torch.nn.functional as F # Not explicitly used, can remove if not needed
import joblib # For loading KNN model
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import cv2
from torchvision import transforms, models
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time # For timing
import pickle # For KNN model loading fallback

# --- Config ---
ARCH_NAME = "vision_transformer"
INFERENCE_TYPE_TAG = "augmented"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Standardized dataset paths
DATASET1_PATH = os.path.join("dataset", "plantvillage", "test")
DATASET2_PATH = os.path.join("dataset", "plantdoc", "test")

# Model paths from the original script (verified as relative)
KNN_MODEL_PATH = os.path.join("models", "glcm_knn", "trained", "glcm_knn.pkl")
# Path for the AugmentedViT model weights
AUGMENTED_MODEL_PATH = os.path.join("models", "vision_transformer", "augmented", "vision_transformer_augmented.pth")
print(f"[INFO] Using Augmented Vision Transformer model path: {AUGMENTED_MODEL_PATH}")
print(f"[INFO] Using KNN model path: {KNN_MODEL_PATH}")

NUM_CLASSES = None # Dynamically determined
CLASS_NAMES = None # Dynamically determined

IMAGE_SIZE = (224, 224) # For Vision Transformer
TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- GLCM extractor (consistent with user's working version) ---
def extract_glcm_features(image_path_or_cv2_img):
    if isinstance(image_path_or_cv2_img, str):
        img_cv2 = cv2.imread(image_path_or_cv2_img)
        if img_cv2 is None:
            try:
                pil_img = Image.open(image_path_or_cv2_img).convert('L')
                gray = np.array(pil_img)
            except Exception as e_pil:
                raise ValueError(f"Image could not be loaded by OpenCV or PIL: {image_path_or_cv2_img}. Error: {e_pil}")
        else:
            gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    elif isinstance(image_path_or_cv2_img, np.ndarray):
        if image_path_or_cv2_img.ndim == 3:
            gray = cv2.cvtColor(image_path_or_cv2_img, cv2.COLOR_BGR2GRAY)
        elif image_path_or_cv2_img.ndim == 2:
            gray = image_path_or_cv2_img
        else:
            raise ValueError("Provided CV2 image is not 2D or 3D.")
    else:
        raise TypeError("Input to extract_glcm_features must be an image path (str) or OpenCV image (np.ndarray).")

    if gray.ndim != 2:
        raise ValueError(f"Grayscale image is not 2D. Shape: {gray.shape}")
    if np.all(gray == gray[0,0]):
        return np.zeros((1,6)) # 6 features

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0],
    ]).reshape(1, -1)

# --- Augmented ViT Model Definition (adapted from original script) ---
class AugmentedViT(nn.Module):
    def __init__(self, num_classes_global_config=9, knn_feature_dim_global_config=9):
        super().__init__()
        # Load ViT backbone.
        # Original script used weights=models.ViT_B_16_Weights.DEFAULT.
        # If MODEL_PATH for AugmentedViT contains the full model state (including backbone),
        # then weights=None is appropriate here. Assuming MODEL_PATH is for the entire AugmentedViT.
        self.backbone = models.vit_b_16(weights=None)
        
        # Modify ViT head to become an identity layer (feature extractor)
        if hasattr(self.backbone, 'heads') and hasattr(self.backbone.heads, 'head'):
            self.feature_dim = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()
        else: # Fallback for other ViT structures if 'heads.head' attribute path changes
             try: # Attempt common alternative for ViT-like models: model.head
                self.feature_dim = self.backbone.head.in_features
                self.backbone.head = nn.Identity()
             except AttributeError:
                raise RuntimeError("Could not adapt ViT model head to nn.Identity(). Check ViT structure.")

        self.knn_feature_dim = knn_feature_dim_global_config

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + self.knn_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes_global_config)
        )

    def forward(self, img_tensor, knn_feature_tensor): # Renamed knn_tensor to knn_feature_tensor
        image_features = self.backbone(img_tensor)
        if knn_feature_tensor.ndim == 1: # Ensure knn_feature_tensor is 2D for cat
            knn_feature_tensor = knn_feature_tensor.unsqueeze(0)
        if image_features.shape[0] != knn_feature_tensor.shape[0]:
             raise ValueError(f"Batch size mismatch: image_features {image_features.shape[0]}, knn_vector {knn_feature_tensor.shape[0]}")

        combined = torch.cat((image_features, knn_feature_tensor), dim=1)
        return self.classifier(combined)

    def load_and_prepare_model(self, model_state_dict_path, device_to_use):
        print(f"[INFO] Loading {ARCH_NAME}_{INFERENCE_TYPE_TAG} state_dict from: {model_state_dict_path}")
        try:
            checkpoint = torch.load(model_state_dict_path, map_location=device_to_use)
            if isinstance(checkpoint, dict) and ('model_state_dict' in checkpoint or 'state_dict' in checkpoint):
                key_to_use = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
                self.load_state_dict(checkpoint[key_to_use])
                print(f"[INFO] Loaded {ARCH_NAME}_{INFERENCE_TYPE_TAG} state_dict from checkpoint key '{key_to_use}'.")
            else:
                self.load_state_dict(checkpoint)
                print(f"[INFO] Loaded {ARCH_NAME}_{INFERENCE_TYPE_TAG} state_dict directly (assumed raw).")
        except FileNotFoundError:
            print(f"[ERROR] {ARCH_NAME}_{INFERENCE_TYPE_TAG} state_dict not found at: {model_state_dict_path}")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to load {ARCH_NAME}_{INFERENCE_TYPE_TAG} state_dict: {e}")
            raise

        self.to(device_to_use).eval()

        if device_to_use.type == 'cuda':
            print(f"[INFO] Performing GPU warm-up for {ARCH_NAME}_{INFERENCE_TYPE_TAG} model...")
            try:
                dummy_img_tensor = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device_to_use)
                dummy_knn_tensor = torch.randn(1, self.knn_feature_dim).to(device_to_use)
                with torch.no_grad():
                    for _ in range(5):
                        _ = self.forward(dummy_img_tensor, dummy_knn_tensor)
                torch.cuda.synchronize()
                print(f"[INFO] {ARCH_NAME}_{INFERENCE_TYPE_TAG} GPU warm-up complete.")
            except Exception as e:
                print(f"[WARN] Error during {ARCH_NAME}_{INFERENCE_TYPE_TAG} warm-up: {e}")
        return self

# --- Load KNN Model ---
def load_knn_model(path):
    print(f"[INFO] Loading KNN model from: {path}")
    try:
        knn = joblib.load(path)
        print("[INFO] KNN model loaded successfully using joblib.")
        return knn
    except pickle.UnpicklingError as upe:
         print(f"[WARN] Failed to unpickle KNN model using joblib: {upe}. Trying with pickle module.")
         try:
             with open(path, "rb") as f:
                 knn = pickle.load(f)
             print("[INFO] KNN model loaded successfully using pickle.")
             return knn
         except Exception as e_pickle:
             print(f"[ERROR] Failed to load KNN model with joblib or pickle from {path}: {e_pickle}")
             raise
    except FileNotFoundError:
        print(f"[ERROR] KNN model file not found at: {path}")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to load KNN model with joblib from {path}: {e}. Trying with pickle.")
        try:
             with open(path, "rb") as f:
                 knn = pickle.load(f)
             print("[INFO] KNN model loaded successfully using pickle.")
             return knn
        except Exception as e_pickle_fallback:
            print(f"[ERROR] Failed to load KNN model with joblib or pickle fallback from {path}: {e_pickle_fallback}")
            raise

# --- Image and Label Loader ---
def get_image_paths_and_labels(root_dir, class_names_config_ref):
    image_paths, labels = [], []
    if not os.path.isdir(root_dir):
        raise ValueError(f"Dataset root directory not found: {root_dir}")

    class_subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if not class_subdirs:
        raise ValueError(f"No class subdirectories found in {root_dir}.")

    current_dataset_class_names = class_subdirs
    global NUM_CLASSES, CLASS_NAMES # Allow modification

    if class_names_config_ref is None:
        CLASS_NAMES = current_dataset_class_names
        NUM_CLASSES = len(CLASS_NAMES)
        print(f"[INFO] Determined global NUM_CLASSES = {NUM_CLASSES}, CLASS_NAMES = {CLASS_NAMES} from {root_dir}")
    elif sorted(current_dataset_class_names) != sorted(class_names_config_ref):
        print(f"[WARN] Dataset {root_dir} classes {current_dataset_class_names} differ from global {class_names_config_ref}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}

    for class_dir_name in current_dataset_class_names:
        if class_dir_name not in CLASS_NAMES:
            print(f"[WARN] Directory '{class_dir_name}' in {root_dir} not in global CLASS_NAMES. Skipping.")
            continue
        cls_idx = class_to_idx[class_dir_name]
        cls_path = os.path.join(root_dir, class_dir_name)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(cls_path, fname))
                labels.append(cls_idx)
    if not image_paths:
        raise ValueError(f"No images found in {root_dir} for known classes based on global CLASS_NAMES.")
    print(f"[INFO] Found {len(image_paths)} images for inference in {root_dir} (mapped to {NUM_CLASSES} global classes).")
    return image_paths, labels, current_dataset_class_names

# --- Save Classification Metrics & Confusion Matrix (Standardized) ---
def save_per_dataset_classification_results(report, conf_mat, arch_conf, inf_type_conf, ds_name, report_cls_names):
    output_dir = os.path.join("results", arch_conf, inf_type_conf, ds_name)
    os.makedirs(output_dir, exist_ok=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f"{arch_conf}_{inf_type_conf}_{ds_name}_classification_report.csv"))
    with open(os.path.join(output_dir, f"{arch_conf}_{inf_type_conf}_{ds_name}_classification_report.txt"), 'w') as f:
        f.write(f"Report for {arch_conf} ({inf_type_conf}) on {ds_name}:\n\n{report_df.to_string()}")
    plt.figure(figsize=(max(10, len(report_cls_names)*0.8), max(8, len(report_cls_names)*0.6)))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=report_cls_names, yticklabels=report_cls_names, annot_kws={"size": 8 if len(report_cls_names) > 10 else 10})
    plt.title(f"CM - {arch_conf} ({inf_type_conf}) on {ds_name}", fontsize=10)
    plt.xlabel("Predicted", fontsize=9); plt.ylabel("True", fontsize=9)
    plt.xticks(fontsize=8, rotation=45, ha="right"); plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{arch_conf}_{inf_type_conf}_{ds_name}_confusion_matrix.png"))
    plt.close()
    print(f"[INFO] Per-dataset classification results saved to: {output_dir}")

# --- Save Combined Inference Timing Results (Standardized) ---
def save_combined_inference_times(all_timing_list, arch_conf, inf_type_conf):
    output_dir_combined = os.path.join("results", arch_conf, inf_type_conf)
    os.makedirs(output_dir_combined, exist_ok=True)
    df_timing = pd.DataFrame(all_timing_list)
    df_timing.to_csv(os.path.join(output_dir_combined, "inference_times.csv"), index=False)
    with open(os.path.join(output_dir_combined, "inference_times.txt"), 'w') as f:
        f.write(f"Combined Timing Summary for {arch_conf} - {inf_type_conf}:\n\n{df_timing.to_string(index=False)}")
    print(f"[INFO] Combined inference timing results saved to: {output_dir_combined}")

# --- Inference for a single dataset (Augmented) ---
def run_augmented_inference_on_dataset(dataset_path, dataset_name_key, arch_name_config, inference_type_tag_config, current_augmented_model_path):
    print(f"\n[INFO] Running augmented inference (WITH WARM-UP) on {dataset_name_key}...")

    try:
        image_paths, true_labels_all, _ = get_image_paths_and_labels(dataset_path, CLASS_NAMES)
    except ValueError as e:
        print(f"[ERROR] Could not load images/labels for {dataset_name_key}: {e}. Skipping.")
        return None

    if NUM_CLASSES is None or CLASS_NAMES is None:
        print(f"[ERROR] Global class information not set. Cannot proceed with {dataset_name_key}.")
        return None

    try:
        # knn_feature_dim_global should match the output of knn.predict_proba()
        # which is NUM_CLASSES (number of classes KNN was trained on, assumed to be our global NUM_CLASSES)
        augmented_model_instance = AugmentedViT(num_classes_global_config=NUM_CLASSES, knn_feature_dim_global_config=NUM_CLASSES)
        augmented_model_instance.load_and_prepare_model(current_augmented_model_path, DEVICE)
        
        knn_model = load_knn_model(KNN_MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Could not load models for {dataset_name_key}: {e}")
        return None

    true_labels_for_report, predicted_labels_for_report = [], []
    total_inference_time_seconds, processed_image_count = 0.0, 0

    print(f"[INFO] Starting inference loop for {len(image_paths)} images in {dataset_name_key}...")
    for img_path, current_true_label in tqdm(zip(image_paths, true_labels_all), total=len(true_labels_all), desc=f"Inferring on {dataset_name_key}"):
        try:
            start_image_time = time.perf_counter()

            glcm_feat_np = extract_glcm_features(img_path) # Expects path, returns (1,6)
            knn_proba_np = knn_model.predict_proba(glcm_feat_np).flatten() # Flatten to 1D array
            if len(knn_proba_np) != NUM_CLASSES:
                raise ValueError(f"KNN proba vector length mismatch for {img_path}. Expected {NUM_CLASSES}, Got {len(knn_proba_np)}")

            pil_image = Image.open(img_path).convert("RGB")
            img_tensor_transformed = TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
            knn_proba_tensor = torch.tensor(knn_proba_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                output_logits = augmented_model_instance(img_tensor_transformed, knn_proba_tensor)
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                final_predicted_class = torch.argmax(output_logits, dim=1).item()
            
            end_image_time = time.perf_counter()
            total_inference_time_seconds += (end_image_time - start_image_time)
            predicted_labels_for_report.append(final_predicted_class)
            true_labels_for_report.append(current_true_label)
            processed_image_count += 1
        except UnidentifiedImageError:
            print(f"[WARN] Skipping image {img_path} due to UnidentifiedImageError (PIL).")
        except ValueError as ve:
            print(f"[WARN] Skipping image {img_path} due to ValueError: {ve}")
        except Exception as e:
            print(f"[WARN] Unexpected error processing {img_path}. Skipped. Error: {e}")

    if processed_image_count == 0:
        print(f"[ERROR] No images successfully processed for {dataset_name_key}.")
        return None

    avg_time_per_image = total_inference_time_seconds / processed_image_count
    print(f"\n[INFO] --- TIMING for {dataset_name_key} ({arch_name_config} - {inference_type_tag_config}) ---")
    print(f"[INFO] Images processed: {processed_image_count} / {len(image_paths)}")
    print(f"[INFO] Total inference time (Augmented model WARMED UP): {total_inference_time_seconds:.4f} s")
    print(f"[INFO] Avg time per image: {avg_time_per_image:.6f} s")

    report_labels_indices = np.arange(NUM_CLASSES)
    # Use global CLASS_NAMES for consistent report labels
    report_dict = classification_report(true_labels_for_report, predicted_labels_for_report, target_names=CLASS_NAMES, labels=report_labels_indices, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(true_labels_for_report, predicted_labels_for_report, labels=report_labels_indices)
    save_per_dataset_classification_results(report_dict, conf_mat, arch_name_config, inference_type_tag_config, dataset_name_key, CLASS_NAMES)

    timing_data = {"Inference Type": dataset_name_key, "Total Time (s)": total_inference_time_seconds, "Avg Time per Image (s)": avg_time_per_image}
    return timing_data

# --- Main Execution ---
if __name__ == "__main__":
    print(f"[INFO] Script Start Time: {time.strftime('%Y%m%d-%H%M%S')}")
    print(f"[INFO] Architecture: {ARCH_NAME}")
    print(f"[INFO] Inference Type Tag: {INFERENCE_TYPE_TAG}")
    print(f"[INFO] NOTE: Inference timing WITH explicit model warm-up.")

    datasets_to_process = {
        "plantvillage": DATASET1_PATH,
        "plantdoc": DATASET2_PATH
    }
    all_timing_results_summary = []

    # Check model paths
    if not os.path.exists(AUGMENTED_MODEL_PATH) or not AUGMENTED_MODEL_PATH.lower().endswith((".pth", ".pt")):
        print(f"[FATAL] AUGMENTED_MODEL_PATH ('{AUGMENTED_MODEL_PATH}') invalid or not .pth/.pt. Update path. Exiting.")
        exit()
    if not os.path.exists(KNN_MODEL_PATH):
        print(f"[FATAL] KNN_MODEL_PATH ('{KNN_MODEL_PATH}') not found. Exiting.")
        exit()

    # Initialize class information
    first_valid_ds_path_for_class_init = None
    for path_val in datasets_to_process.values():
        if os.path.exists(path_val) and os.path.isdir(path_val):
            first_valid_ds_path_for_class_init = path_val
            break
    if first_valid_ds_path_for_class_init:
        try:
            print(f"[INFO] Initializing class information from: {first_valid_ds_path_for_class_init}")
            _, _, _ = get_image_paths_and_labels(first_valid_ds_path_for_class_init, None) # Call to set globals
            if NUM_CLASSES is None or CLASS_NAMES is None: # Check if globals were set
                 raise RuntimeError("Class information (NUM_CLASSES, CLASS_NAMES) not set after initial dataset load.")
        except Exception as e:
            print(f"[FATAL] Could not initialize class information from {first_valid_ds_path_for_class_init}. Error: {e}. Exiting.")
            exit()
    else:
        print("[FATAL] No valid dataset paths in datasets_to_process. Cannot determine class information. Exiting.")
        exit()
    
    # Now NUM_CLASSES and CLASS_NAMES are set globally for model loading in the loop

    for dataset_name_key, dataset_actual_path in datasets_to_process.items():
        if not os.path.exists(dataset_actual_path) or not os.path.isdir(dataset_actual_path):
            print(f"[WARN] Dataset path invalid: {dataset_actual_path} for '{dataset_name_key}'. Skipping.")
            continue
        
        timing_result = run_augmented_inference_on_dataset(
            dataset_actual_path,
            dataset_name_key,
            ARCH_NAME,
            INFERENCE_TYPE_TAG,
            AUGMENTED_MODEL_PATH # Pass the path to the AugmentedViT weights
        )
        if timing_result:
            all_timing_results_summary.append(timing_result)

    if all_timing_results_summary:
        save_combined_inference_times(all_timing_results_summary, ARCH_NAME, INFERENCE_TYPE_TAG)
    else:
        print("[WARN] No timing results generated to save in combined file.")

    print(f"\n[INFO] Script End Time: {time.strftime('%Y%m%d-%H%M%S')}")
    print("[INFO] All operations complete.")