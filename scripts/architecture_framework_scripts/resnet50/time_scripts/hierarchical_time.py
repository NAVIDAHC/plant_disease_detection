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
import time # Added for timing

# --- Config ---
ARCH_NAME = "resnet50"
INFERENCE_TYPE_TAG = "hierarchical" # Defines the 'inference_type' part of the path

NUM_CLASSES = 9
HEALTHY_CLASS_INDEX = 4 # Assuming 'Healthy' is the 5th class (index 4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [ # Ensure this order matches your model's training
    "Bacterial Spot", "Cercospora Leaf Spot", "Common Rust", "Early Blight",
    "Healthy", "Late Blight", "Northern Leaf Blight",
    "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]
DL_MODEL_PATH = "models/resnet50/trained/resnet50_best_model.pth"
KNN_MODEL_PATH = "models/glcm_knn/trained/glcm_knn.pkl"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load ResNet50 ---
def load_dl_model(): # Renamed for clarity
    model = models.resnet50(weights=None) # Using weights=None based on original, adjust if using pretrained for backbone
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(DL_MODEL_PATH, map_location=DEVICE))
    model_dl = model.to(DEVICE).eval()

    # Warm-up for the DL model
    if DEVICE.type == 'cuda':
        print("[INFO] Performing GPU warm-up for DL model...")
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
            with torch.no_grad():
                for _ in range(5): # Multiple warm-up inferences
                    _ = model_dl(dummy_input)
            torch.cuda.synchronize()
            print("[INFO] DL model GPU warm-up complete.")
        except Exception as e:
            print(f"[WARN] Error during DL model warm-up: {e}")
    return model_dl

# --- GLCM Feature Extractor ---
def extract_glcm_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        # Try to load with PIL as a fallback if cv2 fails (e.g. for non-standard paths or formats)
        try:
            pil_img = Image.open(image_path).convert('L') # Convert to grayscale
            gray = np.array(pil_img)
        except Exception as e_pil:
            raise ValueError(f"Image could not be loaded by OpenCV or PIL: {image_path}. OpenCV error (if any): N/A, PIL error: {e_pil}")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if gray.ndim != 2: # Ensure it's 2D
        raise ValueError(f"Grayscale image is not 2D. Path: {image_path}, Shape: {gray.shape}")
    if np.all(gray == gray[0,0]): # Check for blank image
         print(f"[WARN] Image {image_path} appears to be blank or uniform, GLCM might be non-informative.")
         # Return zeros or handle as appropriate, here returning zeros for 6 features
         # This avoids error with graycomatrix on uniform images
         return np.zeros((1,6))


    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ]).reshape(1, -1)

# --- Image Loader ---
def get_image_paths_and_labels(root_dir):
    image_paths, labels = [], []
    # Ensure CLASS_NAMES aligns with the directory names for correct label indexing
    # If directories are 0_ClassName, 1_ClassName, etc., parse them.
    # For simplicity, assuming directory names match CLASS_NAMES or can be mapped.
    # A robust way is to sort directories and assign indices, then map to CLASS_NAMES if needed.
    
    class_subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if not class_subdirs:
        raise ValueError(f"No subdirectories found in {root_dir}. Check dataset structure.")

    # Create a mapping from directory name to index based on CLASS_NAMES
    # This assumes your subdirectories under root_dir are named exactly as in CLASS_NAMES
    try:
        class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}
    except TypeError: # Should not happen if CLASS_NAMES is defined as a list
        print("[ERROR] CLASS_NAMES is not defined correctly.")
        raise

    actual_found_classes = []
    for class_dir_name in class_subdirs:
        if class_dir_name not in CLASS_NAMES:
            print(f"[WARN] Directory '{class_dir_name}' in dataset not found in configured CLASS_NAMES. Skipping.")
            continue
        actual_found_classes.append(class_dir_name)
        cls_idx = class_to_idx[class_dir_name]
        cls_path = os.path.join(root_dir, class_dir_name)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(cls_path, fname))
                labels.append(cls_idx)
    
    if not image_paths:
        raise ValueError(f"No images found in {root_dir} for known classes. Check dataset structure and CLASS_NAMES.")
    
    print(f"[INFO] Found {len(image_paths)} images in {len(actual_found_classes)} classes from {root_dir}.")
    return image_paths, labels

# --- Save Classification Metrics and Confusion Matrix (per dataset_type) ---
def save_per_dataset_classification_results(report, conf_mat, arch_name_config, inference_type_tag_config, dataset_type_name):
    # Path: /results/{architecture}/{inference_type}/{dataset_type}/
    output_dir = os.path.join("results", arch_name_config, inference_type_tag_config, dataset_type_name)
    os.makedirs(output_dir, exist_ok=True)

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f"{arch_name_config}_{inference_type_tag_config}_{dataset_type_name}_classification_report.csv"))
    with open(os.path.join(output_dir, f"{arch_name_config}_{inference_type_tag_config}_{dataset_type_name}_classification_report.txt"), 'w') as f:
        f.write(f"Classification Report for {arch_name_config} ({inference_type_tag_config}) on {dataset_type_name}:\n\n")
        f.write(report_df.to_string())

    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 8})
    plt.title(f"Confusion Matrix - {arch_name_config} ({inference_type_tag_config}) on {dataset_type_name}", fontsize=10)
    plt.xlabel("Predicted", fontsize=9)
    plt.ylabel("Actual", fontsize=9)
    plt.xticks(fontsize=8, rotation=45, ha="right")
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{arch_name_config}_{inference_type_tag_config}_{dataset_type_name}_confusion_matrix.png"))
    plt.close()
    print(f"[INFO] Per-dataset classification results saved to: {output_dir}")


# --- Save Combined Inference Timing Results ---
def save_combined_inference_times(all_datasets_timing_data_list, arch_name_config, inference_type_tag_config):
    # Path: /results/{architecture}/{inference_type}/
    output_dir_for_combined_times = os.path.join("results", arch_name_config, inference_type_tag_config)
    os.makedirs(output_dir_for_combined_times, exist_ok=True)

    df_timing = pd.DataFrame(all_datasets_timing_data_list)
    csv_path = os.path.join(output_dir_for_combined_times, "inference_times.csv")
    df_timing.to_csv(csv_path, index=False)

    txt_path = os.path.join(output_dir_for_combined_times, "inference_times.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Combined Inference Timing Summary for {arch_name_config} - {inference_type_tag_config}:\n\n")
        f.write(df_timing.to_string(index=False))
    print(f"[INFO] Combined inference timing results saved to: {output_dir_for_combined_times}")


# --- Inference for a single dataset ---
def run_hierarchical_inference_on_dataset(dataset_path, dataset_name, arch_name_config, inference_type_tag_config):
    print(f"\n[INFO] Running hierarchical inference on {dataset_name} (Arch: {arch_name_config}, Type: {inference_type_tag_config})...")
    
    try:
        image_paths, true_labels = get_image_paths_and_labels(dataset_path)
    except ValueError as e:
        print(f"[ERROR] Could not load images/labels for {dataset_name}: {e}. Skipping this dataset.")
        return None # Return None if dataset loading fails

    dl_model = load_dl_model() # Load DL model (includes warm-up)
    try:
        knn_model = joblib.load(KNN_MODEL_PATH)
    except FileNotFoundError:
        print(f"[ERROR] KNN model not found at {KNN_MODEL_PATH}. Cannot proceed with hierarchical inference.")
        return None
    except Exception as e:
        print(f"[ERROR] Could not load KNN model: {e}")
        return None


    predicted_labels = []
    total_inference_time_seconds = 0.0
    processed_image_count = 0

    # Check if HEALTHY_CLASS_INDEX is valid
    if not (0 <= HEALTHY_CLASS_INDEX < NUM_CLASSES):
        print(f"[ERROR] HEALTHY_CLASS_INDEX ({HEALTHY_CLASS_INDEX}) is out of bounds for NUM_CLASSES ({NUM_CLASSES}).")
        return None


    print(f"[INFO] Starting inference loop for {len(image_paths)} images in {dataset_name}...")
    for img_path, true_label in tqdm(zip(image_paths, true_labels), total=len(true_labels), desc=f"Inferring on {dataset_name}"):
        try:
            start_image_time = time.perf_counter() # Time the whole hierarchical step per image

            # Stage 1: GLCM + KNN
            glcm_features = extract_glcm_features(img_path)
            knn_predicted_class = knn_model.predict(glcm_features)[0]

            final_predicted_class = -1 # Initialize
            if knn_predicted_class == HEALTHY_CLASS_INDEX:
                final_predicted_class = knn_predicted_class
            else:
                # Stage 2: ResNet50 for non-healthy predicted by KNN
                pil_image = Image.open(img_path).convert("RGB")
                input_tensor = TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    dl_output = dl_model(input_tensor)
                    final_predicted_class = torch.argmax(dl_output, dim=1).item()
            
            end_image_time = time.perf_counter()
            total_inference_time_seconds += (end_image_time - start_image_time)
            
            predicted_labels.append(final_predicted_class)
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
    print(f"[INFO] Total inference time: {total_inference_time_seconds:.4f} seconds")
    print(f"[INFO] Average time per image: {avg_time_per_image:.6f} seconds")

    # Generate and save classification report and confusion matrix
    # Ensure true_labels corresponds to successfully processed images if skips occurred
    # For simplicity here, we assume if an image is skipped, it's not in y_pred, so we need to align y_true.
    # However, the current loop appends to y_pred only on success.
    # If an exception occurs *before* append, true_label for that image is implicitly skipped.
    # For a more robust alignment, one would collect true labels only for successfully processed images.
    # Current approach: report on images where prediction was successful.
    
    # Make sure predicted_labels and the corresponding true_labels are aligned.
    # If processing skips images, true_labels used for report should only be for those successfully processed.
    # The current logic appends to predicted_labels on success. True_labels for report should match this.
    # For this script, let's assume all image_paths are attempted, and if one fails, its true_label is implicitly not part of the final report
    # This is slightly complex if we want to report on a subset.
    # The simplest is to ensure y_true for the report has the same length as y_pred.
    # The current tqdm iterates through true_labels. If an image is skipped, its true_label isn't added to a temporary list.
    # Let's adjust to collect true_labels only for successful predictions.
    
    y_true_for_report = []
    y_pred_for_report = []
    
    # Re-iterate to align true labels for images that were successfully processed
    # This is inefficient. A better way is to build y_true_for_report inside the processing loop.
    # Let's modify the loop:
    
    # Re-initializing for clarity of the improved logic:
    true_labels_for_report = []
    predicted_labels_from_loop = [] # This will replace `predicted_labels`
    total_inference_time_seconds = 0.0 # Reset for recalculation with improved logic
    processed_image_count = 0

    print(f"[INFO] Starting (corrected) inference loop for {len(image_paths)} images in {dataset_name}...")
    for img_path, true_label_for_current_img in tqdm(zip(image_paths, true_labels), total=len(true_labels), desc=f"Inferring (corrected) on {dataset_name}"):
        try:
            start_image_time = time.perf_counter()

            glcm_features = extract_glcm_features(img_path)
            knn_predicted_class = knn_model.predict(glcm_features)[0]

            final_predicted_class_for_current_img = -1
            if knn_predicted_class == HEALTHY_CLASS_INDEX:
                final_predicted_class_for_current_img = knn_predicted_class
            else:
                pil_image = Image.open(img_path).convert("RGB")
                input_tensor = TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    dl_output = dl_model(input_tensor)
                    final_predicted_class_for_current_img = torch.argmax(dl_output, dim=1).item()
            
            end_image_time = time.perf_counter()
            
            # If successful up to here, record everything
            total_inference_time_seconds += (end_image_time - start_image_time)
            predicted_labels_from_loop.append(final_predicted_class_for_current_img)
            true_labels_for_report.append(true_label_for_current_img) # Align true label
            processed_image_count += 1

        except ValueError as ve:
            print(f"[WARN] Skipping image {img_path} due to ValueError: {ve}")
        except Exception as e:
            print(f"[WARN] An unexpected error occurred processing {img_path}. Skipped. Error: {e}")

    if processed_image_count == 0:
        print(f"[ERROR] Corrected loop: No images were successfully processed for {dataset_name}.")
        return None # No data to report or time

    avg_time_per_image = total_inference_time_seconds / processed_image_count

    # Now use `true_labels_for_report` and `predicted_labels_from_loop`
    report_dict = classification_report(true_labels_for_report, predicted_labels_from_loop, target_names=CLASS_NAMES, labels=np.arange(NUM_CLASSES), output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(true_labels_for_report, predicted_labels_from_loop, labels=np.arange(NUM_CLASSES))

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

    # Define dataset paths and their user-friendly names
    # These names will be used as {dataset_type} in paths and "Inference Type" in the summary CSV/TXT
    datasets_to_process = {
        "plantvillage": "dataset/plantvillage/test",
        "plantdoc": "dataset/plantdoc/test"
        # Add more datasets here if needed
    }

    all_timing_results_summary = []

    for dataset_name_key, dataset_actual_path in datasets_to_process.items():
        if not os.path.exists(dataset_actual_path):
            print(f"[WARN] Dataset path not found: {dataset_actual_path} for '{dataset_name_key}'. Skipping.")
            continue
        
        timing_result = run_hierarchical_inference_on_dataset(
            dataset_actual_path,
            dataset_name_key, # Use the key (e.g., "plantvillage") as the {dataset_type}
            ARCH_NAME,
            INFERENCE_TYPE_TAG
        )
        if timing_result: # Only append if processing was successful
            all_timing_results_summary.append(timing_result)

    if all_timing_results_summary:
        save_combined_inference_times(all_timing_results_summary, ARCH_NAME, INFERENCE_TYPE_TAG)
    else:
        print("[WARN] No timing results were generated from any dataset to save in the combined file.")

    print(f"\n[INFO] Script End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("[INFO] All operations complete.")