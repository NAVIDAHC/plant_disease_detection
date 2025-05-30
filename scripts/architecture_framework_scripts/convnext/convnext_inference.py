import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# --- Config ---
DATASET1_PATH = "dataset/plantvillage/test"
DATASET2_PATH = "dataset/plantdoc/test"
MODEL_PATH = "models/convnext/trained/convnext_best_model.pth"
OUTPUT_DIR_BASE = "results/convnext/base_model"
ARCH_NAME = "convnext"
NUM_CLASSES = 9
BATCH_SIZE = 32
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
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# --- DataLoader Loader ---
def load_dual_datasets(path1, path2, batch_size):
    ds1 = datasets.ImageFolder(path1, transform=transform)
    ds2 = datasets.ImageFolder(path2, transform=transform)
    loader1 = DataLoader(ds1, batch_size=batch_size, shuffle=False)
    loader2 = DataLoader(ds2, batch_size=batch_size, shuffle=False)
    return loader1, loader2

# --- Evaluation ---
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    conf_mat = confusion_matrix(all_labels, all_preds)

    return report, conf_mat

# --- Save Results ---
def save_metrics_and_confusion(report, conf_mat, output_dir, prefix="results"):
    os.makedirs(output_dir, exist_ok=True)

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f"{prefix}_inference_result.csv"))
    with open(os.path.join(output_dir, f"{prefix}_inference_result.txt"), 'w') as f:
        f.write(report_df.to_string())

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"))
    plt.close()

# --- Inference Wrapper ---
def run_inference(model, dataloader, dataset_name):
    print(f"[INFO] Running inference on: {dataset_name}")
    report, conf_mat = evaluate_model(model, dataloader)
    output_dir = os.path.join(OUTPUT_DIR_BASE, dataset_name)
    save_metrics_and_confusion(report, conf_mat, output_dir, prefix=f"{ARCH_NAME}")
    print(f"[INFO] Results saved to: {output_dir}")

# --- Main ---
if __name__ == "__main__":
    print("[INFO] Loading model...")
    model = load_model(MODEL_PATH)

    print("[INFO] Loading datasets...")
    loader1, loader2 = load_dual_datasets(DATASET1_PATH, DATASET2_PATH, batch_size=BATCH_SIZE)

    run_inference(model, loader1, "plantvillage")
    run_inference(model, loader2, "plantdoc")
