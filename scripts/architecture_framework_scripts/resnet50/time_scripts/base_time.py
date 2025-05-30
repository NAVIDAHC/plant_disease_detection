import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import time

# --- Config ---
DATASET1_PATH = "dataset/plantvillage/test"
DATASET2_PATH = "dataset/plantdoc/test"
MODEL_PATH = "models/resnet50/trained/resnet50_best_model.pth"

ARCH_NAME = "resnet50"
# This tag will be used for the {inference_type} directory level
# e.g., "base_model", "quantized_model", "pruned_model"
INFERENCE_TYPE_TAG = "base_model"

NUM_CLASSES = 9
BATCH_SIZE = 32 # You can adjust this
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "Bacterial Spot", "Cercospora Leaf Spot", "Common Rust", "Early Blight",
    "Healthy", "Late Blight", "Northern Leaf Blight",
    "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Model Loader ---
def load_model(model_path):
    model = models.resnet50(weights=None) # Original script used weights=None
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# --- DataLoader Loader ---
def load_dual_datasets(path1, path2, batch_size):
    ds1 = datasets.ImageFolder(path1, transform=transform)
    ds2 = datasets.ImageFolder(path2, transform=transform)
    # Using num_workers and pin_memory for potentially faster loading on GPU
    loader1 = DataLoader(ds1, batch_size=batch_size, shuffle=False, num_workers=min(os.cpu_count(), 4), pin_memory=True if DEVICE.type == 'cuda' else False)
    loader2 = DataLoader(ds2, batch_size=batch_size, shuffle=False, num_workers=min(os.cpu_count(), 4), pin_memory=True if DEVICE.type == 'cuda' else False)

    # Get dataset names from their parent directory names for clarity in logs and outputs
    dataset1_name = os.path.basename(os.path.dirname(os.path.normpath(path1))) # e.g., plantvillage
    dataset2_name = os.path.basename(os.path.dirname(os.path.normpath(path2))) # e.g., plantdoc

    print(f"[INFO] Dataset '{dataset1_name}' (from {path1}) contains {len(ds1)} images.")
    print(f"[INFO] Dataset '{dataset2_name}' (from {path2}) contains {len(ds2)} images.")
    return loader1, loader2, len(ds1), len(ds2), dataset1_name, dataset2_name

# --- Evaluation with Timing ---
def evaluate_model_with_timing(model, dataloader, current_dataset_name_for_log):
    model.eval()
    all_preds = []
    all_labels = []
    total_inference_time_seconds = 0.0
    num_images_processed = 0

    # Warm-up run (optional, but good for GPU, especially for the first dataset processed)
    if DEVICE.type == 'cuda':
        print(f"[INFO] Performing GPU warm-up for {current_dataset_name_for_log} dataset...")
        # Create a temporary loader for warm-up to not affect the main one
        temp_warmup_loader = DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, shuffle=False,
                                        num_workers=min(os.cpu_count(),2), pin_memory=True) # Light loader for warmup
        for _ in range(min(5, len(temp_warmup_loader))): # Warm up for a few batches
            try:
                dummy_inputs, _ = next(iter(temp_warmup_loader))
                dummy_inputs = dummy_inputs.to(DEVICE)
                with torch.no_grad():
                    _ = model(dummy_inputs)
                torch.cuda.synchronize() # Ensure warm-up op is complete
            except StopIteration:
                break
        del temp_warmup_loader # Clean up
        print(f"[INFO] GPU warm-up for {current_dataset_name_for_log} complete.")

    print(f"[INFO] Starting timed evaluation for {current_dataset_name_for_log}...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            if DEVICE.type == 'cuda':
                torch.cuda.synchronize() # Ensure readiness before timing
            start_time_batch = time.perf_counter()

            outputs = model(inputs) # Core inference step

            if DEVICE.type == 'cuda':
                torch.cuda.synchronize() # Ensure inference is complete before stopping timer
            end_time_batch = time.perf_counter()

            total_inference_time_seconds += (end_time_batch - start_time_batch)
            num_images_processed += inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(all_labels, all_preds)

    return report, conf_mat, total_inference_time_seconds, num_images_processed

# --- Save Classification Metrics and Confusion Matrix (per dataset_type) ---
def save_per_dataset_classification_results(report, conf_mat, arch_name_config, inference_type_tag_config, dataset_type_name):
    # Path: /results/{architecture}/{inference_type}/{dataset_type}/
    output_dir = os.path.join("results", arch_name_config, inference_type_tag_config, dataset_type_name)
    os.makedirs(output_dir, exist_ok=True)

    report_df = pd.DataFrame(report).transpose()
    # Filenames include arch_name and dataset_type for clarity, though path is already specific
    report_df.to_csv(os.path.join(output_dir, f"{arch_name_config}_{dataset_type_name}_classification_report.csv"))
    with open(os.path.join(output_dir, f"{arch_name_config}_{dataset_type_name}_classification_report.txt"), 'w') as f:
        f.write(f"Classification Report for {arch_name_config} on {dataset_type_name} ({inference_type_tag_config}):\n\n")
        f.write(report_df.to_string())

    plt.figure(figsize=(12, 10)) # Adjusted for better layout with many class names
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 8}) # smaller annotation font
    plt.title(f"Confusion Matrix - {arch_name_config} on {dataset_type_name} ({inference_type_tag_config})", fontsize=10)
    plt.xlabel("Predicted", fontsize=9)
    plt.ylabel("Actual", fontsize=9)
    plt.xticks(fontsize=8, rotation=45, ha="right") # Rotate for readability
    plt.yticks(fontsize=8)
    plt.tight_layout() # Adjust layout
    plt.savefig(os.path.join(output_dir, f"{arch_name_config}_{dataset_type_name}_confusion_matrix.png"))
    plt.close()
    print(f"[INFO] Per-dataset classification results saved to: {output_dir}")

# --- Save Combined Inference Timing Results ---
def save_combined_inference_times(all_datasets_timing_data_list, arch_name_config, inference_type_tag_config):
    # Path: /results/{architecture}/{inference_type}/
    output_dir_for_combined_times = os.path.join("results", arch_name_config, inference_type_tag_config)
    os.makedirs(output_dir_for_combined_times, exist_ok=True)

    df_timing = pd.DataFrame(all_datasets_timing_data_list)

    # CSV: Inference Type,Total Time (s),Avg Time per Image (s)
    # Note: "Inference Type" column in the CSV refers to the dataset_type (e.g., plantvillage)
    csv_path = os.path.join(output_dir_for_combined_times, "inference_times.csv") # Fixed filename as per your request
    df_timing.to_csv(csv_path, index=False)

    txt_path = os.path.join(output_dir_for_combined_times, "inference_times.txt") # Fixed filename as per your request
    with open(txt_path, 'w') as f:
        f.write(f"Combined Inference Timing Summary for {arch_name_config} - {inference_type_tag_config}:\n\n")
        f.write(df_timing.to_string(index=False))

    print(f"[INFO] Combined inference timing results saved to: {output_dir_for_combined_times}")

# --- Main Inference Function per Dataset ---
def run_and_evaluate_one_dataset(model, dataloader, total_images_in_dataset, dataset_name_for_processing,
                                 arch_name_from_config, inference_type_tag_from_config):
    print(f"\n[INFO] Processing Dataset: {dataset_name_for_processing} (Arch: {arch_name_from_config}, Type: {inference_type_tag_from_config})")

    report, conf_mat, total_time, num_actually_processed = evaluate_model_with_timing(model, dataloader, dataset_name_for_processing)

    avg_time_per_image = 0
    if num_actually_processed > 0:
        avg_time_per_image = total_time / num_actually_processed
    else:
        print(f"[WARNING] No images were processed for {dataset_name_for_processing}. Check dataset and dataloader.")

    if num_actually_processed != total_images_in_dataset:
        print(f"[WARNING] For {dataset_name_for_processing}: Number of processed images ({num_actually_processed}) "
              f"does not match total images in dataset ({total_images_in_dataset}). "
              f"Average time is based on the processed count.")

    print(f"[INFO] --- TIMING RESULTS for {dataset_name_for_processing} ---")
    print(f"[INFO] Total images processed: {num_actually_processed}")
    print(f"[INFO] Total inference time: {total_time:.4f} seconds")
    print(f"[INFO] Average time per image: {avg_time_per_image:.6f} seconds")

    # Save the detailed classification results for this specific dataset
    save_per_dataset_classification_results(report, conf_mat, arch_name_from_config,
                                            inference_type_tag_from_config, dataset_name_for_processing)

    # Prepare data for the combined timing file
    timing_data_for_this_dataset = {
        "Inference Type": dataset_name_for_processing, # This will be 'plantvillage' or 'plantdoc'
        "Total Time (s)": total_time,
        "Avg Time per Image (s)": avg_time_per_image
    }
    return timing_data_for_this_dataset

# --- Main Execution ---
if __name__ == "__main__":
    print(f"[INFO] Script Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Architecture: {ARCH_NAME}")
    print(f"[INFO] Inference Type Tag: {INFERENCE_TYPE_TAG}") # e.g. base_model

    print("\n[INFO] Loading model...")
    model = load_model(MODEL_PATH)

    print("\n[INFO] Loading datasets...")
    loader_ds1, loader_ds2, len_ds1, len_ds2, name_ds1, name_ds2 = load_dual_datasets(
        DATASET1_PATH, DATASET2_PATH, batch_size=BATCH_SIZE
    )

    all_datasets_timing_summary_list = []

    # Process first dataset (e.g., PlantVillage)
    timing_ds1 = run_and_evaluate_one_dataset(model, loader_ds1, len_ds1, name_ds1, ARCH_NAME, INFERENCE_TYPE_TAG)
    all_datasets_timing_summary_list.append(timing_ds1)

    # Process second dataset (e.g., PlantDoc)
    timing_ds2 = run_and_evaluate_one_dataset(model, loader_ds2, len_ds2, name_ds2, ARCH_NAME, INFERENCE_TYPE_TAG)
    all_datasets_timing_summary_list.append(timing_ds2)

    # Save the combined inference timing information (for all datasets under this ARCH_NAME and INFERENCE_TYPE_TAG)
    if all_datasets_timing_summary_list:
        save_combined_inference_times(all_datasets_timing_summary_list, ARCH_NAME, INFERENCE_TYPE_TAG)
    else:
        print("[WARNING] No timing results were generated to save in the combined file.")

    print(f"\n[INFO] Script End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("[INFO] All operations complete.")