import subprocess
import time
import csv

# Mapping of inference labels to script paths
inference_scripts = {
    "base": "scripts/architecture_framework_scripts/swin_transformer/swin_transformer_infrence.py",
    "hierarchical": "scripts/architecture_framework_scripts/swin_transformer/swin_transformer_hierarchical_inference.py",
    "augmented": "scripts/architecture_framework_scripts/swin_transformer/swin_transformer_augmented_inference.py",
    "stacking": "scripts/architecture_framework_scripts/swin_transformer/swin_transformer_stacking_inference.py"
}


# Approximate test set sizes
dataset_sizes = {
    "plantvillage": 464,
    "plantdoc": 442
}

# This assumes each script handles both datasets
results = []

for inference_type, script in inference_scripts.items():
    print(f"\n[INFO] Running inference type: {inference_type.upper()}")
    start_time = time.time()

    # Run the script
    subprocess.run(["python", script], check=True)

    end_time = time.time()
    total_time = end_time - start_time

    # Average time assuming both datasets run (you could break it down further if needed)
    total_images = dataset_sizes["plantvillage"] + dataset_sizes["plantdoc"]
    avg_time = total_time / total_images

    results.append([inference_type, total_time, avg_time])
    print(f"[DONE] {inference_type.upper()} → Total: {total_time:.2f}s | Avg/Image: {avg_time:.4f}s")

# Write results to CSV
with open("results/swin_transformer/all_inference_times.csv", mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Inference Type", "Total Time (s)", "Avg Time per Image (s)"])
    writer.writerows(results)

print("\n[INFO] All timings saved to: results/swin_transformer/all_inference_times.csv")
