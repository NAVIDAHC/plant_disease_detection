import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import time # Added for timing
import numpy as np # Added for metrics labels

# --- Config ---
ARCH_NAME = "vision_transformer"
INFERENCE_TYPE_TAG = "base_model"

# Relative paths
DATASET1_PATH = os.path.join("dataset", "plantvillage", "test")
DATASET2_PATH = os.path.join("dataset", "plantdoc", "test")
# Note the typo "trainsformer" in the original path, kept here to match user's file.
MODEL_PATH = os.path.join("models", "vision_transformer", "trained", "vision_transformer_best_model.pth")
print(f"[INFO] Using ViT Model Path: {MODEL_PATH}")


NUM_CLASSES = None # Dynamically determined
CLASS_NAMES = None # Dynamically determined

BATCH_SIZE = 32 # From original script
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Transform (from original script) ---
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), # ViT typically uses 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # Standard ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

# --- Model Loader with Warm-up ---
def load_vit_model_with_warmup(model_weights_path, num_classes_config, device_to_use):
    print(f"[INFO] Loading {ARCH_NAME} model (WITH WARM-UP) for {num_classes_config} classes...")
    # Load architecture. Set weights=None if MODEL_PATH contains the full model.
    # Original script used models.ViT_B_16_Weights.DEFAULT then loaded state_dict.
    # If MODEL_PATH is a full fine-tuned checkpoint, weights=None is more appropriate.
    model = models.vit_b_16(weights=None)

    # Modify the head for the number of classes
    if hasattr(model, 'heads') and hasattr(model.heads, 'head') and isinstance(model.heads.head, nn.Linear):
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes_config)
    else:
        raise RuntimeError(f"Could not modify head of {ARCH_NAME}. Expected model.heads.head to be nn.Linear.")

    try:
        checkpoint = torch.load(model_weights_path, map_location=device_to_use)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
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
            dummy_batch_size = min(BATCH_SIZE, 4, num_classes_config if num_classes_config is not None else 4)
            dummy_input = torch.randn(dummy_batch_size, 3, 224, 224).to(device_to_use)
            with torch.no_grad():
                for _ in range(5):
                    _ = model(dummy_input)
            torch.cuda.synchronize()
            print(f"[INFO] {ARCH_NAME} GPU warm-up complete.")
        except Exception as e:
            print(f"[WARN] Error during {ARCH_NAME} warm-up: {e}")
    print(f"[INFO] {ARCH_NAME} model loaded and warmed up.")
    return model

# --- Dataset Loader ---
def load_dataset_for_inference(dataset_path, batch_size_config, transform_config):
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Dataset path is not a valid directory: {dataset_path}")

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform_config)

    if not dataset.classes:
         raise ValueError(f"No classes found in dataset directory: {dataset_path}. Check structure.")
    if len(dataset) == 0:
         raise ValueError(f"No images found in dataset directory: {dataset_path}")

    global NUM_CLASSES, CLASS_NAMES # Allow modification of global variables
    if NUM_CLASSES is None: # First time loading a dataset
        NUM_CLASSES = len(dataset.classes)
        CLASS_NAMES = list(dataset.classes)
        print(f"[INFO] Determined NUM_CLASSES = {NUM_CLASSES} and CLASS_NAMES = {CLASS_NAMES} from dataset: {dataset_path}")
    elif len(dataset.classes) != NUM_CLASSES or sorted(dataset.classes) != sorted(CLASS_NAMES):
        print(f"[WARN] Dataset at {dataset_path} has {len(dataset.classes)} classes ({dataset.classes}), "
              f"which differs from previously determined {NUM_CLASSES} classes ({CLASS_NAMES}). "
              "Ensure consistency for combined reports or that model is adapted per dataset.")

    dataloader = DataLoader(dataset,
                            batch_size=batch_size_config,
                            shuffle=False,
                            num_workers=min(os.cpu_count(), 2),
                            pin_memory=True if DEVICE.type == 'cuda' else False)
    print(f"[INFO] Loaded dataset from {dataset_path} with {len(dataset)} images in {len(dataset.classes)} classes.")
    return dataloader, len(dataset), list(dataset.classes)


# --- Evaluation with Timing (Batch Processing) ---
def evaluate_model_with_timing(model, dataloader, device_to_use, current_dataset_name_for_log, current_class_names_list):
    model.eval()
    all_preds_list = []
    all_labels_list = []
    total_inference_time_seconds = 0.0
    num_images_processed = 0

    print(f"[INFO] Starting timed evaluation for {current_dataset_name_for_log}...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating {current_dataset_name_for_log}"):
            images, labels = images.to(device_to_use), labels.to(device_to_use)

            if device_to_use.type == 'cuda': torch.cuda.synchronize()
            start_time_batch = time.perf_counter()
            outputs = model(images) # Core inference step
            if device_to_use.type == 'cuda': torch.cuda.synchronize()
            end_time_batch = time.perf_counter()

            total_inference_time_seconds += (end_time_batch - start_time_batch)
            num_images_processed += images.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds_list.extend(predicted.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())

    if num_images_processed == 0:
        print(f"[ERROR] No images were processed in the evaluation loop for {current_dataset_name_for_log}")
        return None, None, 0, 0

    if len(all_labels_list) != len(all_preds_list):
         print(f"[ERROR] Mismatch between labels ({len(all_labels_list)}) and predictions ({len(all_preds_list)}).")
         return None, None, 0, num_images_processed
    
    num_actual_classes = len(current_class_names_list)
    report_labels_indices = np.arange(num_actual_classes)

    report = classification_report(all_labels_list, all_preds_list, target_names=current_class_names_list, labels=report_labels_indices, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(all_labels_list, all_preds_list, labels=report_labels_indices)

    return report, conf_mat, total_inference_time_seconds, num_images_processed

# --- Save Classification Metrics and Confusion Matrix (Standardized) ---
def save_per_dataset_classification_results(report, conf_mat, arch_conf, inf_type_conf, ds_name, report_cls_names):
    output_dir = os.path.join("results", arch_conf, inf_type_conf, ds_name)
    os.makedirs(output_dir, exist_ok=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f"{arch_conf}_{inf_type_conf}_{ds_name}_classification_report.csv"))
    with open(os.path.join(output_dir, f"{arch_conf}_{inf_type_conf}_{ds_name}_classification_report.txt"), 'w') as f:
        f.write(f"Classification Report for {arch_conf} ({inf_type_conf}) on {ds_name}:\n\n{report_df.to_string()}")
    plt.figure(figsize=(max(10, len(report_cls_names)*0.8), max(8, len(report_cls_names)*0.6)))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=report_cls_names, yticklabels=report_cls_names, annot_kws={"size": 8 if len(report_cls_names) > 10 else 10})
    plt.title(f"Confusion Matrix - {arch_conf} ({inf_type_conf}) on {ds_name}", fontsize=10)
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
        f.write(f"Combined Inference Timing Summary for {arch_conf} - {inf_type_conf}:\n\n{df_timing.to_string(index=False)}")
    print(f"[INFO] Combined inference timing results saved to: {output_dir_combined}")

# --- Inference Runner for a single dataset ---
def run_base_inference_on_dataset(dataset_actual_path, dataset_name_key, arch_name_config, inference_type_tag_config, model_weights_path):
    print(f"\n[INFO] Processing dataset: {dataset_name_key} (Arch: {arch_name_config}, Type: {inference_type_tag_config})...")

    try:
        dataloader, total_images_in_dataset, ds_specific_class_names = load_dataset_for_inference(dataset_actual_path, BATCH_SIZE, TRANSFORM)
        report_class_names_for_this_dataset = CLASS_NAMES if CLASS_NAMES else ds_specific_class_names
    except ValueError as e:
        print(f"[ERROR] Failed to load dataset '{dataset_name_key}': {e}. Skipping.")
        return None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred loading dataset '{dataset_name_key}': {e}. Skipping.")
        return None

    try:
        if NUM_CLASSES is None:
             raise RuntimeError("NUM_CLASSES has not been determined before model loading.")
        model_to_eval = load_vit_model_with_warmup(model_weights_path, NUM_CLASSES, DEVICE)
    except FileNotFoundError:
        print(f"[ERROR] Cannot proceed: Model weights not found at {model_weights_path}")
        return None
    except RuntimeError as e:
        print(f"[ERROR] Cannot proceed: Failed to load model {model_weights_path}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred loading model {model_weights_path}: {e}. Skipping dataset.")
        return None

    eval_results = evaluate_model_with_timing(model_to_eval, dataloader, DEVICE, dataset_name_key, report_class_names_for_this_dataset)
    if eval_results is None or eval_results[0] is None:
         print(f"[ERROR] Model evaluation failed for dataset {dataset_name_key}. Skipping.")
         return None
    report_dict, conf_mat, total_time_seconds, num_images_actually_processed = eval_results

    if num_images_actually_processed == 0:
        print(f"[ERROR] Evaluation processed 0 images for {dataset_name_key}.")
        return None

    avg_time_per_image = total_time_seconds / num_images_actually_processed if num_images_actually_processed > 0 else 0
    if num_images_actually_processed != total_images_in_dataset:
        print(f"[WARNING] For {dataset_name_key}: Processed images ({num_images_actually_processed}) != total images in dataset ({total_images_in_dataset}).")

    print(f"\n[INFO] --- TIMING RESULTS for {dataset_name_key} ({arch_name_config} - {inference_type_tag_config}) ---")
    print(f"[INFO] Total images processed: {num_images_actually_processed}")
    print(f"[INFO] Total pure inference time (model forward pass): {total_time_seconds:.4f} seconds")
    print(f"[INFO] Average pure inference time per image: {avg_time_per_image:.6f} seconds")

    save_per_dataset_classification_results(report_dict, conf_mat, arch_name_config, inference_type_tag_config, dataset_name_key, report_class_names_for_this_dataset)

    timing_data = {
        "Inference Type": dataset_name_key,
        "Total Time (s)": total_time_seconds,
        "Avg Time per Image (s)": avg_time_per_image
    }
    return timing_data

# --- Main Execution ---
if __name__ == "__main__":
    print(f"[INFO] Script Start Time: {time.strftime('%Y%m%d-%H%M%S')}")
    print(f"[INFO] Architecture: {ARCH_NAME}")
    print(f"[INFO] Inference Type Tag: {INFERENCE_TYPE_TAG}")
    print("[INFO] NOTE: Inference timing will be performed WITH explicit model warm-up.")

    datasets_to_process = {
        "plantvillage": DATASET1_PATH,
        "plantdoc": DATASET2_PATH
    }
    all_timing_results_summary = []

    if not os.path.exists(MODEL_PATH) or not MODEL_PATH.lower().endswith((".pth", ".pt")):
        print(f"[FATAL] MODEL_PATH ('{MODEL_PATH}') is invalid or not a .pth/.pt file. Exiting.")
        exit()

    first_valid_ds_path = None
    for path_val in datasets_to_process.values():
        if os.path.exists(path_val) and os.path.isdir(path_val):
            first_valid_ds_path = path_val
            break
    if first_valid_ds_path:
        try:
            print(f"[INFO] Initializing class information from first valid dataset: {first_valid_ds_path}")
            _, _, _ = load_dataset_for_inference(first_valid_ds_path, BATCH_SIZE, TRANSFORM)
            if NUM_CLASSES is None or CLASS_NAMES is None:
                 raise RuntimeError("Class information not set after initial dataset load.")
        except Exception as e:
            print(f"[FATAL] Could not initialize class information from {first_valid_ds_path}. Error: {e}. Exiting.")
            exit()
    else:
        print("[FATAL] No valid dataset paths found. Cannot determine class information. Exiting.")
        exit()
    
    for dataset_name_key, dataset_actual_path in datasets_to_process.items():
        if not os.path.exists(dataset_actual_path) or not os.path.isdir(dataset_actual_path):
            print(f"[WARN] Dataset path not found or invalid: {dataset_actual_path} for '{dataset_name_key}'. Skipping.")
            continue

        timing_result = run_base_inference_on_dataset(
            dataset_actual_path,
            dataset_name_key,
            ARCH_NAME,
            INFERENCE_TYPE_TAG,
            MODEL_PATH
        )
        if timing_result:
            all_timing_results_summary.append(timing_result)

    if all_timing_results_summary:
        save_combined_inference_times(all_timing_results_summary, ARCH_NAME, INFERENCE_TYPE_TAG)
    else:
        print("[WARN] No timing results were generated from any dataset to save in the combined file.")

    print(f"\n[INFO] Script End Time: {time.strftime('%Y%m%d-%H%M%S')}")
    print("[INFO] All operations complete.")