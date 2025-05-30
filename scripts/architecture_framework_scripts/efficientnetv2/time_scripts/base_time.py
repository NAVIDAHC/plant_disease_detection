import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from PIL import UnidentifiedImageError, Image # Added Image
from tqdm import tqdm # <<<<<<<<<<<< IMPORT ADDED HERE

# --- Config ---
ARCH_NAME = "efficientnetv2" # Architecture name for pathing
INFERENCE_TYPE_TAG = "base_model" # Type of inference for pathing

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Relative paths for datasets
DATASET1_PATH = os.path.join("dataset", "plantvillage", "test")
DATASET2_PATH = os.path.join("dataset", "plantdoc", "test")

# Relative path for the model (as specified by user)
MODEL_PATH = os.path.join("models", "efficientnetv2", "trained", "efficientnetv2_best_model.pth")

NUM_CLASSES = None # Will be determined from the first loaded dataset
CLASS_NAMES = None # Will be determined from the first loaded dataset

BATCH_SIZE = 32

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Custom Safe Image Loader ---
def safe_image_loader(path: str):
    try:
        return datasets.folder.default_loader(path)
    except (FileNotFoundError, UnidentifiedImageError) as e:
        # print(f"⚠️ Skipping missing/corrupt image: {path}. Error: {e}") # Verbose
        return None
    except Exception as e:
        # print(f"⚠️ Unexpected error loading image: {path}. Error: {e}. Skipping.") # Verbose
        return None

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=safe_image_loader, is_valid_file=None):
        super().__init__(root, transform=transform, target_transform=target_transform, loader=loader, is_valid_file=is_valid_file)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        original_index = index
        retries = 0
        max_retries = len(self.samples)

        while sample is None and retries < max_retries :
            index = (index + 1) % len(self.samples)
            path, target = self.samples[index]
            sample = self.loader(path)
            retries +=1
            if index == original_index and retries > 1:
                 raise RuntimeError(f"All images in the dataset, starting from index {original_index}, failed to load after {retries} retries.")
        if sample is None:
            raise RuntimeError(f"Could not load a valid image for index {original_index} after multiple retries.")

        if self.transform is not None: sample = self.transform(sample)
        if self.target_transform is not None: target = self.target_transform(target)
        return sample, target

# --- Model Loader with Warm-up ---
def load_efficientnet_model_with_warmup(model_weights_path, num_classes_config, device_to_use):
    print(f"[INFO] Loading EfficientNetV2-S model (WITH WARM-UP) for {num_classes_config} classes...")
    model = models.efficientnet_v2_s(weights=None)

    if isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 1 and isinstance(model.classifier[1], nn.Linear):
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes_config)
    else:
        print("[WARN] EfficientNetV2 classifier structure not as expected. Replacing model.classifier directly.")
        try:
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, num_classes_config)
        except AttributeError:
            raise RuntimeError("Could not determine in_features or replace classifier for EfficientNetV2-S.")

    try:
        checkpoint = torch.load(model_weights_path, map_location=device_to_use)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[INFO] Loaded model state_dict from checkpoint key 'model_state_dict' in {model_weights_path}.")
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"[INFO] Loaded model state_dict from checkpoint key 'state_dict' in {model_weights_path}.")
        else:
             model.load_state_dict(checkpoint)
             print(f"[INFO] Loaded model state_dict directly from {model_weights_path} (assumed raw state_dict).")
    except FileNotFoundError:
         print(f"[ERROR] Model weights file not found at: {model_weights_path}")
         raise
    except Exception as e:
         print(f"[ERROR] Failed to load model state_dict from {model_weights_path}: {e}")
         raise

    model.to(device_to_use).eval()

    if device_to_use.type == 'cuda':
        print("[INFO] Performing GPU warm-up for EfficientNetV2-S model...")
        try:
            dummy_input = torch.randn(min(BATCH_SIZE, 4), 3, 224, 224).to(device_to_use)
            with torch.no_grad():
                for _ in range(5): _ = model(dummy_input)
            torch.cuda.synchronize()
            print("[INFO] EfficientNetV2-S GPU warm-up complete.")
        except Exception as e: print(f"[WARN] Error during EfficientNetV2-S warm-up: {e}")
    print("[INFO] EfficientNetV2-S model loaded and warmed up.")
    return model

# --- Dataset Loader ---
def load_dataset_for_inference(dataset_path, batch_size_config, transform_config):
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Dataset path is not a valid directory: {dataset_path}")
    dataset = CustomImageFolder(root=dataset_path, transform=transform_config)
    if not dataset.classes:
         raise ValueError(f"No classes found in {dataset_path}.")
    if len(dataset.samples) == 0:
         raise ValueError(f"No image samples found in {dataset_path}")

    global NUM_CLASSES, CLASS_NAMES
    if NUM_CLASSES is None:
        NUM_CLASSES = len(dataset.classes)
        CLASS_NAMES = list(dataset.classes)
        print(f"[INFO] Determined NUM_CLASSES = {NUM_CLASSES}, CLASS_NAMES = {CLASS_NAMES} from {dataset_path}")
    elif len(dataset.classes) != NUM_CLASSES or sorted(dataset.classes) != sorted(CLASS_NAMES):
        print(f"[WARN] Dataset {dataset_path} classes ({len(dataset.classes)}, {dataset.classes}) "
              f"differ from global ({NUM_CLASSES}, {CLASS_NAMES}).")

    dataloader = DataLoader(dataset, batch_size=batch_size_config, shuffle=False,
                            num_workers=min(os.cpu_count(), 2),
                            pin_memory=True if DEVICE.type == 'cuda' else False)
    print(f"[INFO] Loaded dataset from {dataset_path} with {len(dataset.samples)} samples, {len(dataset.classes)} classes.")
    return dataloader, len(dataset.samples), list(dataset.classes)

# --- Evaluation with Timing ---
def evaluate_model_with_timing(model, dataloader, device_to_use, ds_name_log, current_class_names_list):
    model.eval()
    all_preds_list, all_labels_list = [], []
    total_inference_time_seconds = 0.0
    num_images_processed = 0

    print(f"[INFO] Starting timed evaluation for {ds_name_log}...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating {ds_name_log}"): # tqdm added
            images, labels = images.to(device_to_use), labels.to(device_to_use)
            if device_to_use.type == 'cuda': torch.cuda.synchronize()
            start_time_batch = time.perf_counter()
            outputs = model(images)
            if device_to_use.type == 'cuda': torch.cuda.synchronize()
            end_time_batch = time.perf_counter()

            total_inference_time_seconds += (end_time_batch - start_time_batch)
            num_images_processed += images.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds_list.extend(predicted.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())

    if num_images_processed == 0:
        print(f"[ERROR] No images processed for {ds_name_log}")
        return None, None, 0, 0
    
    num_actual_classes = len(current_class_names_list)
    report = classification_report(all_labels_list, all_preds_list, target_names=current_class_names_list, labels=np.arange(num_actual_classes), output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(all_labels_list, all_preds_list, labels=np.arange(num_actual_classes))
    return report, conf_mat, total_inference_time_seconds, num_images_processed

# --- Save Per-Dataset Classification Results ---
def save_per_dataset_classification_results(report, conf_mat, arch_conf, inf_type_conf, ds_name, current_cls_names):
    output_dir = os.path.join("results", arch_conf, inf_type_conf, ds_name)
    os.makedirs(output_dir, exist_ok=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f"{arch_conf}_{inf_type_conf}_{ds_name}_classification_report.csv"))
    with open(os.path.join(output_dir, f"{arch_conf}_{inf_type_conf}_{ds_name}_classification_report.txt"), 'w') as f:
        f.write(f"Report for {arch_conf} ({inf_type_conf}) on {ds_name}:\n\n{report_df.to_string()}")
    plt.figure(figsize=(max(10, len(current_cls_names)*0.8), max(8, len(current_cls_names)*0.6)))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=current_cls_names, yticklabels=current_cls_names, annot_kws={"size": 8 if len(current_cls_names) > 10 else 10})
    plt.title(f"Confusion Matrix - {arch_conf} ({inf_type_conf}) on {ds_name}", fontsize=10)
    plt.xlabel("Predicted", fontsize=9); plt.ylabel("True", fontsize=9)
    plt.xticks(fontsize=8, rotation=45, ha="right"); plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{arch_conf}_{inf_type_conf}_{ds_name}_confusion_matrix.png"))
    plt.close()
    print(f"[INFO] Per-dataset classification results saved to: {output_dir}")

# --- Save Combined Inference Timing Results ---
def save_combined_inference_times(all_timing_list, arch_conf, inf_type_conf):
    output_dir_combined = os.path.join("results", arch_conf, inf_type_conf)
    os.makedirs(output_dir_combined, exist_ok=True)
    df_timing = pd.DataFrame(all_timing_list)
    df_timing.to_csv(os.path.join(output_dir_combined, "inference_times.csv"), index=False)
    with open(os.path.join(output_dir_combined, "inference_times.txt"), 'w') as f:
        f.write(f"Combined Timing Summary for {arch_conf} - {inf_type_conf}:\n\n{df_timing.to_string(index=False)}")
    print(f"[INFO] Combined inference timing results saved to: {output_dir_combined}")

# --- Inference Runner for a single dataset ---
def run_base_inference_on_dataset(ds_actual_path, ds_name_key, arch_conf, inf_type_conf, model_w_path):
    print(f"\n[INFO] Processing dataset: {ds_name_key} (Arch: {arch_conf}, Type: {inf_type_conf})...")
    current_num_classes = NUM_CLASSES
    current_cls_names = CLASS_NAMES

    try:
        dataloader, total_images, ds_specific_cls_names = load_dataset_for_inference(ds_actual_path, BATCH_SIZE, TRANSFORM)
        if sorted(ds_specific_cls_names) != sorted(current_cls_names):
             print(f"[WARN] Dataset '{ds_name_key}' classes {ds_specific_cls_names} differ from global {current_cls_names}.")
        # Use dataset-specific class names for reporting if they differ, but model is fixed by global NUM_CLASSES
        report_class_names = ds_specific_cls_names
    except Exception as e:
        print(f"[ERROR] Failed to load dataset '{ds_name_key}': {e}. Skipping.")
        return None

    try:
        if current_num_classes is None:
             raise RuntimeError("NUM_CLASSES not determined before model loading.")
        model_to_eval = load_efficientnet_model_with_warmup(model_w_path, current_num_classes, DEVICE)
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_w_path}: {e}. Skipping dataset.")
        return None

    eval_results = evaluate_model_with_timing(model_to_eval, dataloader, DEVICE, ds_name_key, report_class_names)
    if eval_results is None or eval_results[0] is None:
         print(f"[ERROR] Model evaluation failed for {ds_name_key}. Skipping.")
         return None
    report_dict, conf_mat, total_time_s, num_imgs_processed = eval_results

    if num_imgs_processed == 0:
        print(f"[ERROR] Evaluation processed 0 images for {ds_name_key}.")
        return None

    avg_time_per_img = total_time_s / num_imgs_processed if num_imgs_processed > 0 else 0
    if num_imgs_processed != total_images:
        print(f"[WARN] For {ds_name_key}: Processed images ({num_imgs_processed}) != total in dataset ({total_images}).")

    print(f"\n[INFO] --- TIMING for {ds_name_key} ({arch_conf} - {inf_type_conf}) ---")
    print(f"[INFO] Images processed: {num_imgs_processed}")
    print(f"[INFO] Total inference time (model forward pass): {total_time_s:.4f} s")
    print(f"[INFO] Avg inference time per image: {avg_time_per_img:.6f} s")

    save_per_dataset_classification_results(report_dict, conf_mat, arch_conf, inf_type_conf, ds_name_key, report_class_names)
    timing_data = {"Inference Type": ds_name_key, "Total Time (s)": total_time_s, "Avg Time per Image (s)": avg_time_per_img}
    return timing_data

# --- Main Execution ---
if __name__ == "__main__":
    print(f"[INFO] Script Start Time: {time.strftime('%Y%m%d-%H%M%S')}")
    print(f"[INFO] Architecture: {ARCH_NAME}")
    print(f"[INFO] Inference Type Tag: {INFERENCE_TYPE_TAG}")
    print("[INFO] NOTE: Inference timing WITH explicit model warm-up.")

    datasets_to_process = {
        "plantvillage": DATASET1_PATH,
        "plantdoc": DATASET2_PATH
    }
    all_timing_results_summary = []

    if not os.path.exists(MODEL_PATH) or not MODEL_PATH.lower().endswith((".pth", ".pt")):
        print(f"[FATAL] MODEL_PATH ('{MODEL_PATH}') invalid or not .pth/.pt. Update path. Exiting.")
        exit()

    first_valid_ds_path = None
    for path_val in datasets_to_process.values():
        if os.path.exists(path_val) and os.path.isdir(path_val):
            first_valid_ds_path = path_val
            break
    if first_valid_ds_path:
        try:
            print(f"[INFO] Initializing class info from: {first_valid_ds_path}")
            _, _, _ = load_dataset_for_inference(first_valid_ds_path, BATCH_SIZE, TRANSFORM) # Sets globals
            if NUM_CLASSES is None or CLASS_NAMES is None:
                 raise RuntimeError("Class info not set after initial dataset load.")
        except Exception as e:
            print(f"[FATAL] Could not init class info from {first_valid_ds_path}. Error: {e}. Exiting.")
            exit()
    else:
        print("[FATAL] No valid dataset paths in datasets_to_process. Cannot init class info. Exiting.")
        exit()

    for dataset_name, dataset_path in datasets_to_process.items():
        if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
            print(f"[WARN] Dataset path invalid: {dataset_path} for '{dataset_name}'. Skipping.")
            continue
        timing_result = run_base_inference_on_dataset(dataset_path, dataset_name, ARCH_NAME, INFERENCE_TYPE_TAG, MODEL_PATH)
        if timing_result:
            all_timing_results_summary.append(timing_result)

    if all_timing_results_summary:
        save_combined_inference_times(all_timing_results_summary, ARCH_NAME, INFERENCE_TYPE_TAG)
    else:
        print("[WARN] No timing results generated to save in combined file.")

    print(f"\n[INFO] Script End Time: {time.strftime('%Y%m%d-%H%M%S')}")
    print("[INFO] All operations complete.")