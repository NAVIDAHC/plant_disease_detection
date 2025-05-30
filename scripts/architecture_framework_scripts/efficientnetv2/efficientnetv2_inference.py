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
from PIL import UnidentifiedImageError

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to data directories
DATA_DIR = r"C:\Users\User\Desktop\ivan files\plant_disease_detection\dataset\PlantVillage"
TEST_DIR = os.path.join(DATA_DIR, "test")
PLANTDOC_TEST_DIR = r"C:\Users\User\Desktop\ivan files\plant_disease_detection\dataset\Plantdoc\Test"

# Define transformations (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom safe image loader that skips missing files
def safe_image_loader(path):
    try:
        return datasets.folder.default_loader(path)
    except (FileNotFoundError, UnidentifiedImageError):
        print(f"⚠️ Skipping missing/corrupt image: {path}")
        return None  # Mark for filtering

class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if sample is None:
            return self.__getitem__((index + 1) % len(self.samples))  # Skip to next valid image
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, target

# Load test datasets
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
plantdoc_test_dataset = CustomImageFolder(root=PLANTDOC_TEST_DIR, transform=transform, loader=safe_image_loader)

# Define dataloaders
batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
plantdoc_test_loader = DataLoader(plantdoc_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Load the trained model
MODEL_PATH = r"C:\Users\User\Desktop\ivan files\plant_disease_detection\models\architectures_data augmented\efficientnet\efficientnetv2_plantvillage_fold1_epochs.csv"
model = models.efficientnet_v2_s()
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(test_dataset.classes))  

# Load model weights
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# Function to plot and save confusion matrix
def plot_confusion_matrix(cm, class_names, dataset_name, timestamp):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {dataset_name}")
    
    cm_filename = f"confusion_matrix_{dataset_name.replace(' ', '_')}_{timestamp}.png"
    plt.savefig(cm_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved as {cm_filename}")

# Evaluation function with confusion matrix
def evaluate_model(model, dataloader, dataset_name):
    all_preds, all_labels = [], []
    batch_times = []

    start_time = time.time()  # Start total inference time

    with torch.no_grad():
        for images, labels in dataloader:
            batch_start = time.time()  
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            batch_time = time.time() - batch_start  
            batch_times.append(batch_time)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_time = time.time() - start_time  
    avg_batch_time = np.mean(batch_times)  

    # Compute classification metrics
    class_report = classification_report(
        all_labels, all_preds, 
        target_names=test_dataset.classes, 
        output_dict=True,
        zero_division=0  # Prevents warning for undefined precision
    )

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plot_confusion_matrix(cm, test_dataset.classes, dataset_name, timestamp)

    # Convert report to DataFrame
    df = pd.DataFrame(class_report).transpose()
    df["Timestamp"] = timestamp

    # Save as CSV
    csv_filename = f"efficientnetv2_inference_results_{dataset_name.replace(' ', '_')}_{timestamp}.csv"
    df.to_csv(csv_filename, index=True)

    # Save as TXT (includes timing details)
    txt_filename = f"efficientnetv2_inference_results_{dataset_name.replace(' ', '_')}_{timestamp}.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(f"🔹 Inference Results - {timestamp}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total Inference Time: {total_time:.4f} seconds\n")
        f.write(f"Average Inference Time per Batch: {avg_batch_time:.4f} seconds\n")
        f.write(df.to_string())

    print(f"Inference completed for {dataset_name}. Total Time: {total_time:.2f}s, Avg Batch Time: {avg_batch_time:.4f}s")
    print(f"Results saved as CSV and TXT.")

# Run inference
evaluate_model(model, test_loader, "PlantVillage_Test")
evaluate_model(model, plantdoc_test_loader, "PlantDoc_Test")

print("Inference complete. Results saved.")
