import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from skimage.feature import graycomatrix, graycoprops
from datetime import datetime
import pandas as pd

# ---------------------------- SETTINGS ---------------------------- #
GLCM_KNN_MODEL_PATH = "models\glcm_knn_models\glcm_knn_data_modified.pkl"  # Path to trained GLCM+KNN model
EFFNET_MODEL_PATH = r"C:\Users\User\Desktop\ivan files\plant_disease_detection\models\architectures_data augmented\efficientnet\efficientnet.pth"  # Path to trained EfficientNetV2 model
PLANTVILLAGE_TEST_DIR = "dataset/PlantVillage/Test"  # PlantVillage test dataset
PLANTDOC_TEST_DIR = "dataset/Plantdoc/Test"  # PlantDoc test dataset
GLCM_DISTANCES = [1, 2, 3]  # Distance settings for GLCM
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles for GLCM
IMAGE_SIZE = (224, 224)  # Input size for EfficientNetV2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------------------------------------------------ #

def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=GLCM_DISTANCES, angles=GLCM_ANGLES, symmetric=True, normed=True)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        features.extend(graycoprops(glcm, prop).flatten())
    return np.array(features)

def load_knn_model():
    with open(GLCM_KNN_MODEL_PATH, "rb") as f:
        knn = pickle.load(f)
    return knn

def load_efficientnet():
    model = models.efficientnet_v2_s(weights=None)
    checkpoint = torch.load(EFFNET_MODEL_PATH, map_location=DEVICE)
    num_classes = checkpoint['classifier.1.weight'].shape[0]
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)

def run_inference(dataset_dir, dataset_name, knn_model, effnet_model):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f"inference_results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    stage1_preds, stage1_labels = [], []
    stage2_preds, stage2_labels = [], []

    final_labels, final_preds = [], []
    final_report_rows = []

    class_names = sorted(os.listdir(dataset_dir))
    class_map = {name: idx for idx, name in enumerate(class_names)}
    
    total_time = 0
    batch_count = 0
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            start_time = datetime.now()
            features = extract_glcm_features(image).reshape(1, -1)
            stage1_pred = knn_model.predict(features)[0]
            final_label = class_map[class_name]
            final_pred = stage1_pred  # default = Healthy

            # If diseased (not Healthy), use Stage 2 model
            if stage1_pred != class_map.get('Healthy', -1):
                input_tensor = process_image(img_path)
                with torch.no_grad():
                    output = effnet_model(input_tensor)
                    stage2_pred = torch.argmax(output, dim=1).item()
                stage2_preds.append(stage2_pred)
                stage2_labels.append(final_label)
                final_pred = stage2_pred

            # Collect for final output
            stage1_preds.append(stage1_pred)
            stage1_labels.append(final_label)
            final_preds.append(final_pred)
            final_labels.append(final_label)

            final_report_rows.append({
                "Image": img_name,
                "True Label": class_name,
                "Stage 1 Prediction": list(class_map.keys())[list(class_map.values()).index(stage1_pred)],
                "Stage 2 Prediction": list(class_map.keys())[list(class_map.values()).index(final_pred)] if stage1_pred != class_map.get('Healthy', -1) else "N/A",
                "Final Prediction": list(class_map.keys())[list(class_map.values()).index(final_pred)],
                "Correct": final_pred == final_label
            })

            stage1_preds.append(stage1_pred)
            stage1_labels.append(class_map[class_name])
            
            if stage1_pred != class_map.get('Healthy', -1):
                input_tensor = process_image(img_path)
                with torch.no_grad():
                    output = effnet_model(input_tensor)
                    stage2_pred = torch.argmax(output, dim=1).item()
                stage2_preds.append(stage2_pred)
                stage2_labels.append(class_map[class_name])
            
            end_time = datetime.now()
            total_time += (end_time - start_time).total_seconds()
            batch_count += 1
    
    avg_time_per_batch = total_time / batch_count if batch_count > 0 else 0
    
    stage1_df = save_results(stage1_labels, stage1_preds, dataset_name, "stage1", results_dir, total_time, avg_time_per_batch, timestamp)
    stage2_df = save_results(stage2_labels, stage2_preds, dataset_name, "stage2", results_dir, total_time, avg_time_per_batch, timestamp)
    save_combined_results(stage1_df, stage2_df, dataset_name, results_dir, timestamp)

    save_final_results(final_labels, final_preds, final_report_rows, dataset_name, results_dir, total_time, avg_time_per_batch, timestamp)

    

def save_results(true_labels, pred_labels, dataset_name, stage, results_dir, total_time, avg_time, timestamp):
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{dataset_name} - {stage.capitalize()} Confusion Matrix")
    plt.savefig(f"{results_dir}/{dataset_name}_{stage}_confusion_matrix.png")
    plt.close()
    
    report = classification_report(true_labels, pred_labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report["Timestamp"] = timestamp
    df_report.to_csv(f"{results_dir}/{dataset_name}_{stage}_results.csv", index=True)
    
    with open(f"{results_dir}/{dataset_name}_{stage}_results.txt", "w", encoding="utf-8") as f:
        f.write(f"\U0001F539 Inference Results - {timestamp}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total Inference Time: {total_time:.4f} seconds\n")
        f.write(f"Average Inference Time per Batch: {avg_time:.4f} seconds\n")
        f.write(df_report.to_string())
    
    print(f"Saved results in: {results_dir}")
    return df_report

def save_combined_results(stage1_df, stage2_df, dataset_name, results_dir, timestamp):
    # Save combined CSVs
    stage1_csv_path = os.path.join(results_dir, f"{dataset_name}_stage1_report.csv")
    stage2_csv_path = os.path.join(results_dir, f"{dataset_name}_stage2_report.csv")
    stage1_df.to_csv(stage1_csv_path, index=True)
    stage2_df.to_csv(stage2_csv_path, index=True)

    # Save combined TXT
    combined_txt_path = os.path.join(results_dir, f"{dataset_name}_combined_results.txt")
    with open(combined_txt_path, "w", encoding="utf-8") as f:
        f.write(f"📊 Combined Results - {timestamp}\n")
        f.write(f"Dataset: {dataset_name}\n\n")
        f.write("🔹 Stage 1 Classification Report\n")
        f.write(stage1_df.to_string())
        f.write("\n\n")
        f.write("🔹 Stage 2 Classification Report\n")
        f.write(stage2_df.to_string())

    print(f"Saved combined report: {combined_txt_path}")

def save_final_results(true_labels, pred_labels, detailed_rows, dataset_name, results_dir, total_time, avg_time, timestamp):
    # Confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{dataset_name} - Final Decision Confusion Matrix")
    plt.savefig(f"{results_dir}/{dataset_name}_final_confusion_matrix.png")
    plt.close()

    # Classification report
    report = classification_report(true_labels, pred_labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report["Timestamp"] = timestamp
    df_report.to_csv(f"{results_dir}/{dataset_name}_final_results.csv", index=True)

    # Detailed final report
    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_csv(f"{results_dir}/{dataset_name}_final_detailed_predictions.csv", index=False)

    # TXT summary
    with open(f"{results_dir}/{dataset_name}_final_results.txt", "w", encoding="utf-8") as f:
        f.write(f"✅ FINAL Inference Report - {timestamp}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total Inference Time: {total_time:.4f} seconds\n")
        f.write(f"Average Inference Time per Batch: {avg_time:.4f} seconds\n\n")
        f.write(df_report.to_string())

    print(f"📦 Final results saved to: {results_dir}")


def main():
    print("Loading models...")
    knn_model = load_knn_model()
    effnet_model = load_efficientnet()
    
    print("Running inference on PlantVillage test dataset...")
    run_inference(PLANTVILLAGE_TEST_DIR, "PlantVillage_Test", knn_model, effnet_model)
    
    print("Running inference on PlantDoc test dataset...")
    run_inference(PLANTDOC_TEST_DIR, "PlantDoc_Test", knn_model, effnet_model)

if __name__ == "__main__":
    main()
