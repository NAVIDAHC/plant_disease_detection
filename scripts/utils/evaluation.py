import torch
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, dataloader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    conf_mat = confusion_matrix(all_labels, all_preds)

    return report, conf_mat, all_preds, all_labels


def save_metrics_and_confusion(report, conf_mat, output_dir, prefix="results"):
    os.makedirs(output_dir, exist_ok=True)

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f"{prefix}_classification_report.csv"))
    with open(os.path.join(output_dir, f"{prefix}_classification_report.txt"), 'w') as f:
        f.write(report_df.to_string())

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"))
    plt.close()
