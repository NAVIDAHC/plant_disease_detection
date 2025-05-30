import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import time # Added for timing
import numpy as np # Added for labels in metrics

# --- Config ---
ARCH_NAME = "regnet"
INFERENCE_TYPE_TAG = "base_model" # Defines the 'inference_type' part of the path

NUM_CLASSES = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [ # Ensure order matches training and dataset folder names
    "Bacterial Spot", "Cercospora Leaf Spot", "Common Rust", "Early Blight",
    "Healthy", "Late Blight", "Northern Leaf Blight",
    "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]
MODEL_PATH = "models/regnet/trained/regnet_best_model.pth"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), # RegNet typically uses 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # Standard ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])
BATCH_SIZE = 32 # Adjust based on GPU memory

# --- Model Loader with Warm-up ---
def load_regnet_model_with_warmup(model_path, device_to_use): # Renamed and added warm-up
    # Using regnet_y_800mf based on original script
    # Weights=None assuming the state_dict contains the full model weights
    # If only fine-tuned, might need models.RegNet_Y_800MF_Weights.DEFAULT
    model = models.regnet_y_800mf(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES) # Adjust final layer
    try:
        state_dict = torch.load(model_path, map_location=device_to_use)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
         print(f"[ERROR] Model file not found at: {model_path}")
         raise
    except Exception as e:
         print(f"[ERROR] Failed to load model state_dict: {e}")
         raise

    model.to(device_to_use).eval()

    # Warm-up
    if device_to_use.type == 'cuda':
        print("[INFO] Performing GPU warm-up for RegNet model...")
        try:
            # Create a dummy batch
            dummy_input = torch.randn(min(BATCH_SIZE, 4), 3, 224, 224).to(device_to_use) # Smaller batch for warm-up ok
            with torch.no_grad():
                for _ in range(5): # Multiple warm-up inferences
                    _ = model(dummy_input)
            torch.cuda.synchronize()
            print("[INFO] RegNet GPU warm-up complete.")
        except Exception as e:
            print(f"[WARN] Error during RegNet warm-up: {e}")
    return model

# --- Dataset Loader ---
def load_dataset_for_inference(dataset_path, batch_size, transform_config):
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Dataset path is not a valid directory: {dataset_path}")
        
    dataset = datasets.ImageFolder(dataset_path, transform=transform_config)
    if not dataset.classes:
         raise ValueError(f"No classes found in dataset directory: {dataset_path}. Check structure.")
    if len(dataset) == 0:
         raise ValueError(f"No images found in dataset directory: {dataset_path}")

    # Check if dataset classes match configured CLASS_NAMES (optional but recommended)
    if sorted(dataset.classes) != sorted(CLASS_NAMES):
         print(f"[WARN] Dataset classes {sorted(dataset.classes)} do not exactly match configured CLASS_NAMES {sorted(CLASS_NAMES)}. Ensure mapping is correct.")

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False, # No shuffle for inference/evaluation
                            num_workers=min(os.cpu_count(), 4), # Use multiple workers for loading
                            pin_memory=True if DEVICE.type == 'cuda' else False) # Pin memory if using CUDA
    print(f"[INFO] Loaded dataset from {dataset_path} with {len(dataset)} images in {len(dataset.classes)} classes.")
    return dataloader, len(dataset)


# --- Evaluation with Timing (Batch Processing) ---
def evaluate_model_with_timing(model, dataloader, device_to_use, current_dataset_name_for_log):
    model.eval()
    all_preds = []
    all_labels = []
    total_inference_time_seconds = 0.0
    num_images_processed = 0

    # No separate warm-up here as it's done during model loading
    # If dataloader itself needed warm-up (e.g., first few batches slow due to disk IO),
    # you could iterate a few times before the timed loop.

    print(f"[INFO] Starting timed evaluation for {current_dataset_name_for_log}...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating {current_dataset_name_for_log}"):
            inputs, labels = inputs.to(device_to_use), labels.to(device_to_use)

            if device_to_use.type == 'cuda':
                torch.cuda.synchronize() # Ensure data transfer is done
            start_time_batch = time.perf_counter()

            outputs = model(inputs) # Core inference step

            if device_to_use.type == 'cuda':
                torch.cuda.synchronize() # Ensure model forward pass is complete
            end_time_batch = time.perf_counter()

            total_inference_time_seconds += (end_time_batch - start_time_batch)
            num_images_processed += inputs.size(0) # Add number of images in the batch

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Ensure labels for report match the number of predictions made
    if len(all_labels) != len(all_preds):
         print(f"[ERROR] Mismatch between number of labels ({len(all_labels)}) and predictions ({len(all_preds)}). Check evaluation loop.")
         # Handle error appropriately, e.g., return None or raise exception
         return None, None, 0, 0 # Indicate failure

    # Generate metrics using collected labels and predictions
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, labels=np.arange(NUM_CLASSES), output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(all_labels, all_preds, labels=np.arange(NUM_CLASSES))

    return report, conf_mat, total_inference_time_seconds, num_images_processed


# --- Save Classification Metrics and Confusion Matrix (per dataset_type) ---
def save_per_dataset_classification_results(report, conf_mat, arch_name_conf, inf_type_tag_conf, ds_type_name):
    # Reusing the standardized function from previous scripts
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
    # Reusing the standardized function
    output_dir_combined = os.path.join("results", arch_name_conf, inf_type_tag_conf)
    os.makedirs(output_dir_combined, exist_ok=True)
    df_timing = pd.DataFrame(all_datasets_timing_list)
    df_timing.to_csv(os.path.join(output_dir_combined, "inference_times.csv"), index=False)
    with open(os.path.join(output_dir_combined, "inference_times.txt"), 'w') as f:
        f.write(f"Combined Inference Timing Summary for {arch_name_conf} - {inf_type_tag_conf}:\n\n{df_timing.to_string(index=False)}")
    print(f"[INFO] Combined inference timing results saved to: {output_dir_combined}")


# --- Inference Runner for a single dataset ---
def run_base_inference_on_dataset(dataset_path, dataset_name, arch_name_config, inference_type_tag_config):
    print(f"\n[INFO] Running base inference on {dataset_name} (Arch: {arch_name_config}, Type: {inference_type_tag_config})...")

    try:
        dataloader, total_images_in_dataset = load_dataset_for_inference(dataset_path, BATCH_SIZE, TRANSFORM)
    except ValueError as e:
        print(f"[ERROR] Failed to load dataset '{dataset_name}': {e}. Skipping.")
        return None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred loading dataset '{dataset_name}': {e}. Skipping.")
        return None

    # Load model (includes warm-up)
    try:
        model = load_regnet_model_with_warmup(MODEL_PATH, DEVICE)
    except FileNotFoundError:
        print(f"[ERROR] Cannot proceed without model file at {MODEL_PATH}.")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}. Skipping dataset {dataset_name}.")
        return None


    # Evaluate model and get timings
    eval_results = evaluate_model_with_timing(model, dataloader, DEVICE, dataset_name)
    if eval_results is None or eval_results[0] is None: # Check if evaluation failed
         print(f"[ERROR] Model evaluation failed for dataset {dataset_name}. Skipping.")
         return None
         
    report_dict, conf_mat, total_time_seconds, num_images_actually_processed = eval_results


    if num_images_actually_processed == 0:
        print(f"[ERROR] Evaluation processed 0 images for {dataset_name}. Cannot generate report or timings.")
        return None

    avg_time_per_image = total_time_seconds / num_images_actually_processed if num_images_actually_processed > 0 else 0

    if num_images_actually_processed != total_images_in_dataset:
        print(f"[WARNING] For {dataset_name}: Processed images ({num_images_actually_processed}) != total images in dataset ({total_images_in_dataset}). "
              f"This might indicate issues during dataloading or evaluation loop.")

    print(f"\n[INFO] --- TIMING RESULTS for {dataset_name} ({arch_name_config} - {inference_type_tag_config}) ---")
    print(f"[INFO] Total images processed: {num_images_actually_processed}")
    print(f"[INFO] Total pure inference time (model forward pass): {total_time_seconds:.4f} seconds")
    print(f"[INFO] Average pure inference time per image: {avg_time_per_image:.6f} seconds")

    save_per_dataset_classification_results(report_dict, conf_mat, arch_name_config, inference_type_tag_config, dataset_name)

    timing_data = {
        "Inference Type": dataset_name, # Dataset name (e.g., plantvillage)
        "Total Time (s)": total_time_seconds,
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
        # Check dataset path existence before proceeding
        if not os.path.exists(dataset_actual_path) or not os.path.isdir(dataset_actual_path):
            print(f"[WARN] Dataset path not found or invalid: {dataset_actual_path} for '{dataset_name_key}'. Skipping.")
            continue

        timing_result = run_base_inference_on_dataset(
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