import os
import torch
import torch.nn as nn
from torchvision import models, transforms
# Removed 'datasets' from torchvision imports as 'get_image_paths_and_labels' handles custom iteration
from PIL import Image, UnidentifiedImageError
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib # Using joblib for consistency
import cv2
from skimage.feature import graycomatrix, graycoprops
import time
import pickle # For KNN model loading fallback

# --- Config ---
ARCH_NAME = "vision_transformer"
INFERENCE_TYPE_TAG = "hierarchical"

NUM_CLASSES = None # Dynamically determined
HEALTHY_CLASS_INDEX = None # Dynamically determined
HEALTHY_CLASS_NAME = "Healthy" # Define the name of your healthy class
CLASS_NAMES = None # Dynamically determined

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Relative paths (from original script, ensuring os.path.join for consistency)
DL_MODEL_PATH = os.path.join("models", "vision_transformer", "trained", "vision_transformer_best_model.pth")
KNN_MODEL_PATH = os.path.join("models", "glcm_knn", "trained", "glcm_knn.pkl")
print(f"[INFO] Using DL Model Path: {DL_MODEL_PATH}")
print(f"[INFO] Using KNN Model Path: {KNN_MODEL_PATH}")


DATASET1_PATH = os.path.join("dataset", "plantvillage", "test")
DATASET2_PATH = os.path.join("dataset", "plantdoc", "test")

IMAGE_SIZE = (224, 224) # ViT typically uses 224x224 or 384x384
TRANSFORM = transforms.Compose([ # For the DL model
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load DL Model (Vision Transformer WITH Warm-up) ---
def load_vit_model_with_warmup(model_weights_path, num_classes_config, device_to_use):
    print(f"[INFO] Loading {ARCH_NAME} model (WITH WARM-UP) for {num_classes_config} classes...")
    # Assuming MODEL_PATH is a full checkpoint, so weights=None
    # Original script used models.ViT_B_16_Weights.DEFAULT then loaded state_dict.
    # If state_dict is full, weights=None is more appropriate.
    model = models.vit_b_16(weights=None)

    # Modify the head for the number of classes (specific to ViT)
    if hasattr(model, 'heads') and hasattr(model.heads, 'head') and isinstance(model.heads.head, nn.Linear):
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes_config)
    else:
        raise RuntimeError(f"Could not modify head of {ARCH_NAME}. Expected model.heads.head to be nn.Linear.")

    try:
        checkpoint = torch.load(model_weights_path, map_location=device_to_use)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint: # Common alternative
            model.load_state_dict(checkpoint['state_dict'])
        else: # Assume it's a raw state_dict
             model.load_state_dict(checkpoint)
        print(f"[INFO] Loaded {ARCH_NAME} state_dict from {model_weights_path}.")
    except FileNotFoundError:
         print(f"[ERROR] {ARCH_NAME} weights file not found at: {model_weights_path}")
         raise
    except Exception as e:
         print(f"[ERROR] Failed to load {ARCH_NAME} state_dict from {model_weights_path}: {e}")
         raise

    model.to(device_to_use).eval()

    if device_to_use.type == 'cuda':
        print(f"[INFO] Performing GPU warm-up for {ARCH_NAME} model...")
        try:
            dummy_batch_size = min(4, num_classes_config if num_classes_config is not None else 2)
            dummy_input = torch.randn(dummy_batch_size, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device_to_use)
            with torch.no_grad():
                for _ in range(5): _ = model(dummy_input)
            torch.cuda.synchronize()
            print(f"[INFO] {ARCH_NAME} GPU warm-up complete.")
        except Exception as e: print(f"[WARN] Error during {ARCH_NAME} warm-up: {e}")
    print(f"[INFO] {ARCH_NAME} model loaded and warmed up.")
    return model

# --- GLCM Feature Extractor (consistent with working RegNet hierarchical) ---
def extract_glcm_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        try:
            pil_img = Image.open(image_path).convert('L')
            gray = np.array(pil_img)
        except Exception as e_pil:
            raise ValueError(f"Image could not be loaded by OpenCV or PIL: {image_path}. Error: {e_pil}")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if gray.ndim != 2:
        raise ValueError(f"Grayscale image is not 2D. Path: {image_path}, Shape: {gray.shape}")
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

# --- Image and Label Loader (Standardized) ---
def get_image_paths_and_labels(root_dir, class_names_config_ref):
    image_paths, labels = [], []
    if not os.path.isdir(root_dir):
        raise ValueError(f"Dataset root directory not found: {root_dir}")

    class_subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if not class_subdirs:
        raise ValueError(f"No class subdirectories found in {root_dir}.")

    current_dataset_class_names = class_subdirs
    global NUM_CLASSES, CLASS_NAMES, HEALTHY_CLASS_INDEX

    if class_names_config_ref is None: # First dataset
        CLASS_NAMES = current_dataset_class_names
        NUM_CLASSES = len(CLASS_NAMES)
        print(f"[INFO] Determined global NUM_CLASSES = {NUM_CLASSES}, CLASS_NAMES = {CLASS_NAMES} from {root_dir}")
        try:
            HEALTHY_CLASS_INDEX = CLASS_NAMES.index(HEALTHY_CLASS_NAME)
            print(f"[INFO] Determined global HEALTHY_CLASS_INDEX = {HEALTHY_CLASS_INDEX} ('{HEALTHY_CLASS_NAME}')")
        except ValueError:
            print(f"[WARN] '{HEALTHY_CLASS_NAME}' not found in determined CLASS_NAMES: {CLASS_NAMES}. Hierarchical logic might fail.")
            HEALTHY_CLASS_INDEX = -1 # Invalid index
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

# --- Inference for a single dataset (Hierarchical) ---
def run_hierarchical_inference_on_dataset(dataset_path, dataset_name_key, arch_name_config, inference_type_tag_config, current_dl_model_path):
    print(f"\n[INFO] Running hierarchical inference (WITH WARM-UP for DL model) on {dataset_name_key}...")

    try:
        image_paths, true_labels_all, _ = get_image_paths_and_labels(dataset_path, CLASS_NAMES)
    except ValueError as e:
        print(f"[ERROR] Could not load images/labels for {dataset_name_key}: {e}. Skipping.")
        return None
    
    if NUM_CLASSES is None or CLASS_NAMES is None or HEALTHY_CLASS_INDEX is None:
        print(f"[ERROR] Critical class information not set. Exiting for {dataset_name_key}.")
        return None
    if HEALTHY_CLASS_INDEX == -1: # Check if Healthy class was validly found
        print(f"[ERROR] Healthy class '{HEALTHY_CLASS_NAME}' not found in CLASS_NAMES. Cannot proceed with {dataset_name_key}.")
        return None

    try:
        # Load DL model (Vision Transformer) with warm-up
        dl_model = load_vit_model_with_warmup(current_dl_model_path, NUM_CLASSES, DEVICE)
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

            glcm_features_np = extract_glcm_features(img_path) # Takes path
            knn_predicted_class_idx = knn_model.predict(glcm_features_np)[0]

            final_predicted_class_idx = -1
            if knn_predicted_class_idx == HEALTHY_CLASS_INDEX:
                final_predicted_class_idx = knn_predicted_class_idx
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
            else:
                pil_image_for_dl = Image.open(img_path).convert("RGB")
                input_tensor = TRANSFORM(pil_image_for_dl).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    if DEVICE.type == 'cuda': torch.cuda.synchronize()
                    dl_output_logits = dl_model(input_tensor)
                    if DEVICE.type == 'cuda': torch.cuda.synchronize()
                    final_predicted_class_idx = torch.argmax(dl_output_logits, dim=1).item()
            
            end_image_time = time.perf_counter()
            total_inference_time_seconds += (end_image_time - start_image_time)
            predicted_labels_for_report.append(final_predicted_class_idx)
            true_labels_for_report.append(current_true_label)
            processed_image_count += 1
        except UnidentifiedImageError:
            print(f"[WARN] Skipping image {img_path} due to UnidentifiedImageError (PIL).")
        except ValueError as ve: # Catch errors from GLCM or image loading within the loop
            print(f"[WARN] Skipping image {img_path} due to ValueError: {ve}")
        except Exception as e:
            print(f"[WARN] Unexpected error processing {img_path}. Skipped. Error: {e}")

    if processed_image_count == 0:
        print(f"[ERROR] No images successfully processed for {dataset_name_key}.")
        return None

    avg_time_per_image = total_inference_time_seconds / processed_image_count
    print(f"\n[INFO] --- TIMING for {dataset_name_key} ({arch_name_config} - {inference_type_tag_config}) ---")
    print(f"[INFO] Images processed: {processed_image_count} / {len(image_paths)}")
    print(f"[INFO] Total inference time (DL model WARMED UP): {total_inference_time_seconds:.4f} s")
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
    print(f"[INFO] NOTE: Inference timing WITH explicit DL model warm-up.")

    datasets_to_process = {
        "plantvillage": DATASET1_PATH,
        "plantdoc": DATASET2_PATH
    }
    all_timing_results_summary = []

    # Check model paths
    if not os.path.exists(DL_MODEL_PATH) or not DL_MODEL_PATH.lower().endswith((".pth", ".pt")):
        print(f"[FATAL] DL_MODEL_PATH ('{DL_MODEL_PATH}') invalid or not .pth/.pt. Exiting.")
        exit()
    if not os.path.exists(KNN_MODEL_PATH):
        print(f"[FATAL] KNN_MODEL_PATH ('{KNN_MODEL_PATH}') not found. Exiting.")
        exit()

    # Initialize class information from the first valid dataset
    first_valid_ds_path_for_class_init = None
    for path_val in datasets_to_process.values():
        if os.path.exists(path_val) and os.path.isdir(path_val):
            first_valid_ds_path_for_class_init = path_val
            break
    if first_valid_ds_path_for_class_init:
        try:
            print(f"[INFO] Initializing class information from: {first_valid_ds_path_for_class_init}")
            # This call sets global CLASS_NAMES, NUM_CLASSES, HEALTHY_CLASS_INDEX
            _, _, _ = get_image_paths_and_labels(first_valid_ds_path_for_class_init, None)
            if NUM_CLASSES is None or CLASS_NAMES is None or HEALTHY_CLASS_INDEX is None:
                 raise RuntimeError("Class/Healthy index info not set after initial dataset load.")
            if HEALTHY_CLASS_INDEX == -1: # Check if 'Healthy' class was found and index is valid
                print(f"[FATAL] Healthy class name '{HEALTHY_CLASS_NAME}' was not found in the classes of the first dataset ({CLASS_NAMES}). "
                      "Hierarchical logic cannot proceed reliably. Please check dataset class names. Exiting.")
                exit()
        except Exception as e:
            print(f"[FATAL] Could not initialize class information from {first_valid_ds_path_for_class_init}. Error: {e}. Exiting.")
            exit()
    else:
        print("[FATAL] No valid dataset paths in datasets_to_process. Cannot determine class information. Exiting.")
        exit()
    
    # Now global class info (NUM_CLASSES, CLASS_NAMES, HEALTHY_CLASS_INDEX) is set for model loading and logic

    for dataset_name_key, dataset_actual_path in datasets_to_process.items():
        if not os.path.exists(dataset_actual_path) or not os.path.isdir(dataset_actual_path):
            print(f"[WARN] Dataset path invalid: {dataset_actual_path} for '{dataset_name_key}'. Skipping.")
            continue
        
        timing_result = run_hierarchical_inference_on_dataset(
            dataset_actual_path,
            dataset_name_key,
            ARCH_NAME,
            INFERENCE_TYPE_TAG,
            DL_MODEL_PATH # Pass the ViT model path
        )
        if timing_result:
            all_timing_results_summary.append(timing_result)

    if all_timing_results_summary:
        save_combined_inference_times(all_timing_results_summary, ARCH_NAME, INFERENCE_TYPE_TAG)
    else:
        print("[WARN] No timing results generated to save in combined file.")

    print(f"\n[INFO] Script End Time: {time.strftime('%Y%m%d-%H%M%S')}")
    print("[INFO] All operations complete.")