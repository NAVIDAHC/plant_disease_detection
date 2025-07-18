import os
import torch
import torch.nn as nn
from torchvision import models, transforms
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
import time

# --- Config ---
ARCH_NAME = "regnet"
INFERENCE_TYPE_TAG = "hierarchical" # Defines the 'inference_type' part of the path

NUM_CLASSES = 9
HEALTHY_CLASS_INDEX = 4 # Assuming 'Healthy' is the 5th class (index 4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [ # Ensure this order matches your model's training and dataset folder names
    "Bacterial Spot", "Cercospora Leaf Spot", "Common Rust", "Early Blight",
    "Healthy", "Late Blight", "Northern Leaf Blight",
    "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]
DL_MODEL_PATH = "models/regnet/trained/regnet_best_model.pth"
KNN_MODEL_PATH = "models/glcm_knn/trained/glcm_knn.pkl"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load DL Model (WITH Warm-up) ---
def load_regnet_model_with_warmup(model_path, device_to_use): # Renamed to reflect warm-up
    print("[INFO] Loading RegNet model (WITH WARM-UP)...")
    # Using regnet_y_800mf based on original script
    # Weights=None assuming state_dict contains full weights.
    # If fine-tuned from default weights, use: weights=models.RegNet_Y_800MF_Weights.DEFAULT
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

    # Warm-up for the RegNet model
    if device_to_use.type == 'cuda':
        print("[INFO] Performing GPU warm-up for RegNet model...")
        try:
            # Create a dummy batch for warm-up
            # Using a small batch size for warm-up is fine
            dummy_input = torch.randn(min(4, NUM_CLASSES), 3, 224, 224).to(device_to_use)
            with torch.no_grad():
                for _ in range(5): # Multiple warm-up inferences
                    _ = model(dummy_input)
            torch.cuda.synchronize()
            print("[INFO] RegNet GPU warm-up complete.")
        except Exception as e:
            print(f"[WARN] Error during RegNet warm-up: {e}")
    print("[INFO] RegNet model loaded and warmed up.")
    return model

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
        # print(f"[WARN] Image {image_path} appears blank/uniform. GLCM might be non-informative, returning zeros.")
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

# --- Load Image Paths and Labels ---
def get_image_paths_and_labels(root_dir, class_names_config): # Robust version
    image_paths, labels = [], []
    class_subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if not class_subdirs:
        raise ValueError(f"No subdirectories found in {root_dir}.")

    try:
        class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names_config)}
    except TypeError:
        print("[ERROR] CLASS_NAMES is not defined correctly as a list.")
        raise

    actual_found_classes_count = 0
    for class_dir_name in class_subdirs:
        if class_dir_name not in class_names_config:
            # print(f"[WARN] Dir '{class_dir_name}' not in CLASS_NAMES. Skipping.")
            continue
        actual_found_classes_count += 1
        cls_idx = class_to_idx[class_dir_name]
        cls_path = os.path.join(root_dir, class_dir_name)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(cls_path, fname))
                labels.append(cls_idx)

    if not image_paths:
        raise ValueError(f"No images found in {root_dir} for configured classes.")

    print(f"[INFO] Found {len(image_paths)} images in {actual_found_classes_count} recognized classes from {root_dir}.")
    return image_paths, labels

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
def run_hierarchical_inference_on_dataset_with_warmup(dataset_path, dataset_name, arch_name_config, inference_type_tag_config): # Renamed function
    print(f"\n[INFO] Running hierarchical inference (WITH WARM-UP) on {dataset_name} (Arch: {arch_name_config}, Type: {inference_type_tag_config})...")

    try:
        image_paths, true_labels_all = get_image_paths_and_labels(dataset_path, CLASS_NAMES)
    except ValueError as e:
        print(f"[ERROR] Could not load images/labels for {dataset_name}: {e}. Skipping.")
        return None

    # Load models (RegNet WITH warm-up)
    try:
        dl_model = load_regnet_model_with_warmup(DL_MODEL_PATH, DEVICE) # Call the warm-up version
        knn_model = joblib.load(KNN_MODEL_PATH)
    except FileNotFoundError as fnf_err:
        print(f"[ERROR] Model file not found: {fnf_err}. Cannot proceed.")
        return None
    except Exception as e:
        print(f"[ERROR] Could not load models: {e}")
        return None

    true_labels_for_report = []
    predicted_labels_for_report = []
    total_inference_time_seconds = 0.0
    processed_image_count = 0

    # Check if HEALTHY_CLASS_INDEX is valid
    if not (0 <= HEALTHY_CLASS_INDEX < NUM_CLASSES):
        print(f"[ERROR] HEALTHY_CLASS_INDEX ({HEALTHY_CLASS_INDEX}) is out of bounds for NUM_CLASSES ({NUM_CLASSES}).")
        return None

    print(f"[INFO] Starting inference loop for {len(image_paths)} images in {dataset_name}...")
    for img_path, current_true_label in tqdm(zip(image_paths, true_labels_all), total=len(true_labels_all), desc=f"Inferring on {dataset_name}"):
        try:
            # Optional: synchronize before timing each image if desired for extremely fine-grained measurement
            # if DEVICE.type == 'cuda':
            #      torch.cuda.synchronize()
            start_image_time = time.perf_counter() # Time the whole hierarchical step per image

            # Stage 1: GLCM + KNN
            glcm_features = extract_glcm_features(img_path)
            knn_predicted_class = knn_model.predict(glcm_features)[0]

            final_predicted_class = -1 # Initialize
            if knn_predicted_class == HEALTHY_CLASS_INDEX:
                final_predicted_class = knn_predicted_class
                # If on CUDA, ensure any (unlikely) preceding CUDA ops are done before stopping timer
                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
            else:
                # Stage 2: RegNet for non-healthy predicted by KNN
                pil_image = Image.open(img_path).convert("RGB")
                input_tensor = TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                     # No explicit sync needed before model call unless prior CUDA ops exist on this stream
                     dl_output = dl_model(input_tensor)
                     # Sync *after* the model call before reading results or stopping timer for this path
                     if DEVICE.type == 'cuda':
                          torch.cuda.synchronize()
                     final_predicted_class = torch.argmax(dl_output, dim=1).item()

            end_image_time = time.perf_counter()
            total_inference_time_seconds += (end_image_time - start_image_time)

            # Record results if successful
            predicted_labels_for_report.append(final_predicted_class)
            true_labels_for_report.append(current_true_label)
            processed_image_count += 1

        except ValueError as ve: # Catch specific errors from GLCM or image loading
            print(f"[WARN] Skipping image {img_path} due to ValueError: {ve}")
        except Exception as e:
            print(f"[WARN] An unexpected error occurred processing {img_path}. Skipped. Error: {e}")

    if processed_image_count == 0:
        print(f"[ERROR] No images were successfully processed for {dataset_name}. Cannot generate report or timings.")
        return None

    avg_time_per_image = total_inference_time_seconds / processed_image_count if processed_image_count > 0 else 0

    print(f"\n[INFO] --- TIMING RESULTS for {dataset_name} ({arch_name_config} - {inference_type_tag_config}) ---")
    print(f"[INFO] Total images processed: {processed_image_count} / {len(image_paths)}")
    print(f"[INFO] Total inference time (WITH WARM-UP): {total_inference_time_seconds:.4f} seconds")
    print(f"[INFO] Average time per image (WITH WARM-UP): {avg_time_per_image:.6f} seconds")

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

        timing_result = run_hierarchical_inference_on_dataset_with_warmup( # Call the warm-up version
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