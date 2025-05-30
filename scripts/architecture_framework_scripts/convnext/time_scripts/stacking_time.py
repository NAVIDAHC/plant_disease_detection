import os
import torch
import torch.nn as nn
import torch.nn.functional as F # For F.softmax
from torchvision import models, transforms
# 'datasets' module from torchvision is not directly used if get_image_paths_and_labels handles iteration
from PIL import Image, UnidentifiedImageError
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib # For loading scikit-learn models
import cv2
from skimage.feature import graycomatrix, graycoprops
import time
import pickle # For KNN model loading fallback

# --- Config ---
ARCH_NAME = "convnext"
INFERENCE_TYPE_TAG = "stacking"

NUM_CLASSES = None # Dynamically determined
CLASS_NAMES = None # Dynamically determined

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Relative paths (from original script, ensuring os.path.join for consistency)
DL_MODEL_PATH = os.path.join("models", "convnext", "trained", "convnext_best_model.pth")
KNN_MODEL_PATH = os.path.join("models", "glcm_knn", "trained", "glcm_knn.pkl")
META_MODEL_PATH = os.path.join("models", "convnext", "meta_learner", "convnext_meta_learner.pkl")
print(f"[INFO] Using DL Model Path: {DL_MODEL_PATH}")
print(f"[INFO] Using KNN Model Path: {KNN_MODEL_PATH}")
print(f"[INFO] Using Meta Learner Path: {META_MODEL_PATH}")

DATASET1_PATH = os.path.join("dataset", "plantvillage", "test")
DATASET2_PATH = os.path.join("dataset", "plantdoc", "test")

IMAGE_SIZE = (224, 224) # Standard for ConvNeXt
TRANSFORM = transforms.Compose([ # Standard transform for the DL model
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load DL Model (ConvNeXt base model WITH Warm-up) ---
def load_convnext_base_model_with_warmup(model_weights_path, num_classes_config, device_to_use):
    print(f"[INFO] Loading {ARCH_NAME} base model (WITH WARM-UP) for {num_classes_config} classes...")
    # Assuming DL_MODEL_PATH is a full checkpoint, so weights=None
    model = models.convnext_tiny(weights=None)

    # Modify the classifier for ConvNeXt_Tiny
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential) and \
       len(model.classifier) > 2 and isinstance(model.classifier[2], nn.Linear):
        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, num_classes_config)
    else: # Fallback
        try:
            if isinstance(model.classifier, nn.Linear):
                 num_features = model.classifier.in_features
                 model.classifier = nn.Linear(num_features, num_classes_config)
            elif isinstance(model.classifier, nn.Sequential) and isinstance(model.classifier[-1], nn.Linear):
                 num_features = model.classifier[-1].in_features
                 model.classifier[-1] = nn.Linear(num_features, num_classes_config)
            else:
                raise AttributeError("Classifier structure not recognized for direct modification.")
            print(f"[WARN] {ARCH_NAME} classifier structure modified using a fallback approach.")
        except AttributeError as e:
            raise RuntimeError(f"Could not modify classifier of {ARCH_NAME}. Expected model.classifier[2] or similar. Error: {e}")

    try:
        checkpoint = torch.load(model_weights_path, map_location=device_to_use)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
             model.load_state_dict(checkpoint)
        print(f"[INFO] Loaded {ARCH_NAME} base model state_dict from {model_weights_path}.")
    except FileNotFoundError:
         print(f"[ERROR] {ARCH_NAME} base model weights file not found at: {model_weights_path}")
         raise
    except Exception as e:
         print(f"[ERROR] Failed to load {ARCH_NAME} base model state_dict from {model_weights_path}: {e}")
         raise

    model.to(device_to_use).eval()

    if device_to_use.type == 'cuda':
        print(f"[INFO] Performing GPU warm-up for {ARCH_NAME} base model...")
        try:
            dummy_batch_size = min(4, num_classes_config if num_classes_config is not None else 2)
            dummy_input = torch.randn(dummy_batch_size, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device_to_use)
            with torch.no_grad():
                for _ in range(5): _ = model(dummy_input)
            torch.cuda.synchronize()
            print(f"[INFO] {ARCH_NAME} base model GPU warm-up complete.")
        except Exception as e: print(f"[WARN] Error during {ARCH_NAME} base model warm-up: {e}")
    print(f"[INFO] {ARCH_NAME} base model loaded and warmed up.")
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

# --- Load Scikit-learn Models (KNN, Meta-Learner) ---
def load_sklearn_model(path, model_type="Model"):
    print(f"[INFO] Loading {model_type} from: {path}")
    try:
        model = joblib.load(path)
        print(f"[INFO] {model_type} loaded successfully using joblib.")
        return model
    except pickle.UnpicklingError as upe:
         print(f"[WARN] Failed to unpickle {model_type} using joblib: {upe}. Trying with pickle.")
         try:
             with open(path, "rb") as f:
                 model = pickle.load(f)
             print(f"[INFO] {model_type} loaded successfully using pickle.")
             return model
         except Exception as e_pickle:
             print(f"[ERROR] Failed to load {model_type} with joblib or pickle from {path}: {e_pickle}")
             raise
    except FileNotFoundError:
        print(f"[ERROR] {model_type} file not found at: {path}")
        raise
    except Exception as e: # Catch other joblib errors
        print(f"[ERROR] Failed to load {model_type} with joblib from {path}: {e}. Trying with pickle.")
        try:
             with open(path, "rb") as f:
                 model = pickle.load(f)
             print(f"[INFO] {model_type} loaded successfully using pickle.")
             return model
        except Exception as e_pickle_fallback:
            print(f"[ERROR] Failed to load {model_type} with joblib or pickle fallback from {path}: {e_pickle_fallback}")
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

    if class_names_config_ref is None: # First dataset
        CLASS_NAMES = current_dataset_class_names
        NUM_CLASSES = len(CLASS_NAMES)
        print(f"[INFO] Determined global NUM_CLASSES = {NUM_CLASSES}, CLASS_NAMES = {CLASS_NAMES} from {root_dir}")
    elif sorted(current_dataset_class_names) != sorted(class_names_config_ref):
        print(f"[WARN] Dataset {root_dir} classes {current_dataset_class_names} differ from global {class_names_config_ref}.")

    # Use the globally determined CLASS_NAMES for mapping to ensure consistency
    class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}

    for class_dir_name in current_dataset_class_names: # Iterate actual dirs found in current dataset
        if class_dir_name not in CLASS_NAMES: # Check if this dir name is part of the global list
            print(f"[WARN] Directory '{class_dir_name}' in {root_dir} not in globally defined CLASS_NAMES. Skipping its contents.")
            continue
        
        cls_idx = class_to_idx[class_dir_name] # Get index based on global list
        cls_path = os.path.join(root_dir, class_dir_name)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(cls_path, fname))
                labels.append(cls_idx) # Append the globally consistent index
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

# --- Inference for a single dataset (Stacking) ---
def run_stacking_inference_on_dataset(dataset_path, dataset_name_key, arch_name_config, inference_type_tag_config,
                                      current_dl_model_path, current_knn_model_path, current_meta_model_path):
    print(f"\n[INFO] Running stacking inference (WITH WARM-UP for DL model) on {dataset_name_key}...")

    try:
        image_paths, true_labels_all, _ = get_image_paths_and_labels(dataset_path, CLASS_NAMES)
    except ValueError as e:
        print(f"[ERROR] Could not load images/labels for {dataset_name_key}: {e}. Skipping.")
        return None
    
    if NUM_CLASSES is None or CLASS_NAMES is None: # Should be set by first get_image_paths_and_labels call
        print(f"[ERROR] Critical class information (NUM_CLASSES, CLASS_NAMES) not set. Exiting for {dataset_name_key}.")
        return None

    try:
        # Load base DL model (ConvNeXt) with warm-up
        dl_model = load_convnext_base_model_with_warmup(current_dl_model_path, NUM_CLASSES, DEVICE)
        knn_model = load_sklearn_model(current_knn_model_path, "KNN Model")
        meta_model = load_sklearn_model(current_meta_model_path, "Meta-Learner Model")
    except Exception as e:
        print(f"[ERROR] Could not load one or more models for {dataset_name_key}: {e}")
        return None

    true_labels_for_report, predicted_labels_for_report = [], []
    total_inference_time_seconds, processed_image_count = 0.0, 0

    print(f"[INFO] Starting inference loop for {len(image_paths)} images in {dataset_name_key}...")
    for img_path, current_true_label in tqdm(zip(image_paths, true_labels_all), total=len(true_labels_all), desc=f"Inferring on {dataset_name_key}"):
        try:
            start_image_time = time.perf_counter()

            # 1. DL Model Prediction (probabilities)
            pil_image_for_dl = Image.open(img_path).convert("RGB")
            img_tensor_transformed = TRANSFORM(pil_image_for_dl).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                dl_model_output_logits = dl_model(img_tensor_transformed)
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                dl_model_probabilities_np = F.softmax(dl_model_output_logits, dim=1).cpu().numpy().flatten()

            # 2. KNN Model Prediction (probabilities from GLCM features)
            glcm_feat_np = extract_glcm_features(img_path) # Expected shape (1,6)
            knn_model_probabilities_np = knn_model.predict_proba(glcm_feat_np).flatten()

            if len(dl_model_probabilities_np) != NUM_CLASSES or len(knn_model_probabilities_np) != NUM_CLASSES:
                raise ValueError(f"Prob vector length mismatch for {img_path}. DL: {len(dl_model_probabilities_np)}, KNN: {len(knn_model_probabilities_np)}")

            # 3. Combine probabilities for Meta-Learner input
            stacked_features_np = np.concatenate((dl_model_probabilities_np, knn_model_probabilities_np)).reshape(1, -1)
            
            # 4. Meta-Learner Prediction
            final_predicted_class = meta_model.predict(stacked_features_np)[0]
            
            end_image_time = time.perf_counter()
            total_inference_time_seconds += (end_image_time - start_image_time)
            predicted_labels_for_report.append(final_predicted_class)
            true_labels_for_report.append(current_true_label)
            processed_image_count += 1
        except UnidentifiedImageError:
            print(f"[WARN] Skipping image {img_path} due to UnidentifiedImageError (PIL).")
        except ValueError as ve: # Catch errors from GLCM or probability vector issues
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
    if not os.path.exists(META_MODEL_PATH):
        print(f"[FATAL] META_MODEL_PATH ('{META_MODEL_PATH}') not found. Exiting.")
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
        
        timing_result = run_stacking_inference_on_dataset(
            dataset_actual_path,
            dataset_name_key,
            ARCH_NAME,
            INFERENCE_TYPE_TAG,
            DL_MODEL_PATH,      # Path for ConvNeXt
            KNN_MODEL_PATH,
            META_MODEL_PATH
        )
        if timing_result:
            all_timing_results_summary.append(timing_result)

    if all_timing_results_summary:
        save_combined_inference_times(all_timing_results_summary, ARCH_NAME, INFERENCE_TYPE_TAG)
    else:
        print("[WARN] No timing results generated to save in combined file.")

    print(f"\n[INFO] Script End Time: {time.strftime('%Y%m%d-%H%M%S')}")
    print("[INFO] All operations complete.")