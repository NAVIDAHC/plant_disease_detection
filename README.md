# Mitigating Accuracy Loss in Plant Disease Detection: A Comparative Study of Multi-Stage Hybrid Classification Frameworks for Field Conditions

This repository contains code for a comparative study of multi-stage hybrid classification frameworks for plant disease detection. The goal is to improve the accuracy of plant disease classification, especially in raw field images where current popular models often show degraded performance. The project combines traditional image processing (GLCM feature extraction), machine learning (KNN classifier), and various deep learning methods in several multi-stage frameworks.

## Overview

The project investigates and compares three primary hybrid approach frameworks using a variety of base models:

1.  **Sequential (Hierarchical) Approach:**
    * Uses **GLCM + KNN** for initial classification.
    * If an image is classified as diseased by GLCM+KNN, a second-stage deep learning model refines the diagnosis. (This behavior is primarily for the GLCM+KNN specific hierarchical model; other deep learning models might have different hierarchical implementations).
2.  **Augmented Approach:**
    * GLCM features are extracted and used to augment the input to a deep learning model, which then performs the classification.
3.  **Stacking Approach:**
    * Predictions from multiple models (e.g., GLCM+KNN and a deep learning model) are used as input to a meta-learner (Logistic Regression in this project) for the final classification.
4.  **Base Model Approach:**
    * Standard deep learning models are trained and evaluated as a baseline.

This repository includes implementations for the GLCM+KNN classifier and several deep learning architectures (ConvNext, EfficientNetV2, RegNet, ResNet50, Swin Transformer, Vision Transformer) within these frameworks.

## Project Structure
```
Root/ (plant_disease_detection/)
├── dataset/                  # Datasets are not included, download via links below
│   ├── plantdoc/
│   │   ├── test/
│   │   │   ├── [Various disease folders]/
│   │   ├── train/
│   │   │   ├── [Various disease folders]/
│   │   └── val/
│   │       ├── [Various disease folders]/
│   └── plantvillage/
│       ├── test/
│       │   ├── [Various disease folders]/
│       ├── train/
│       │   ├── [Various disease folders]/
│       └── val/
│           ├── [Various disease folders]/
├── models/                   # Directory where trained models will be saved by the user
│   ├── convnext/
│   │   ├── augmented/
│   │   ├── base_model/
│   │   ├── hierarchical/
│   │   ├── stacking_meta_learner/
│   │   └── stacking_base_dl_model/
│   ├── efficientnetv2/       # (similar sub-structure for each model type)
│   ├── glcm_knn/
│   │   └── base_model/
│   ├── regnet/               # (similar sub-structure for each model type)
│   ├── resnet50/             # (similar sub-structure for each model type)
│   ├── swin_transformer/     # (similar sub-structure for each model type)
│   └── vision_transformer/   # (similar sub-structure for each model type)
├── results/                  # Evaluation results (CSV, TXT metrics, and JPG confusion matrices)
│   ├── convnext/
│   │   ├── augmented/ (plantdoc/, plantvillage/)
│   │   ├── base_model/ (plantdoc/, plantvillage/)
│   │   ├── hierarchical/ (plantdoc/, plantvillage/)
│   │   └── stacking/ (plantdoc/, plantvillage/)
│   ├── efficientnetv2/       # (similar sub-structure)
│   ├── glcm_knn/
│   │   └── base_model/ (plantdoc/, plantvillage/)
│   ├── regnet/               # (similar sub-structure)
│   ├── resnet50/             # (similar sub-structure)
│   ├── swin_transformer/     # (similar sub-structure)
│   └── vision_transformer/   # (similar sub-structure)
├── scripts/
│   ├── architecture_framework_scripts/
│   │   ├── convnext/         # Contains training, inference, and timing scripts for ConvNext
│   │   │   ├── time_scripts/ # Contains scripts to measure execution time for ConvNext approaches
│   │   │   │   ├── augmented_time.py
│   │   │   │   ├── base_time.py
│   │   │   │   ├── hierarchical_time.py
│   │   │   │   └── stacking_time.py
│   │   │   ├── convnext_augmented_inference.py
│   │   │   ├── convnext_augmented_training.py
│   │   │   ├── convnext_hierarchical_inference.py
│   │   │   ├── convnext_inference.py
│   │   │   ├── convnext_meta_learner.py
│   │   │   ├── convnext_stacking_inference.py
│   │   │   └── convnext_training.py
│   │   ├── efficientnetv2/   # (similar script structure for EfficientNetV2)
│   │   ├── glcm_knn/         # Contains scripts for GLCM+KNN (e.g., glcm_knn.py, potentially time scripts)
│   │   ├── regnet/           # (similar script structure for RegNet)
│   │   ├── resnet50/         # (similar script structure for ResNet50)
│   │   ├── swin_transformer/ # (similar script structure for Swin Transformer)
│   │   └── vision_transformer/ # (similar script structure for Vision Transformer)
│   └── utils/                # Utility scripts
├── requirements.txt          # Required Python packages
├── folder_structure.txt      # Detailed folder structure (generated by script)
├── README.md                 # This file
└── LICENSE                   # GNU GPLv3 License file
## Prerequisites

-   **Python 3.6+** is required.
-   Install the necessary packages using the `requirements.txt` file.
```
### Requirements

```
numpy>=1.18.0
pandas>=1.0.0
opencv-python>=4.2.0.34
scikit-image>=0.17.0
scikit-learn>=0.22
imbalanced-learn>=0.7.0
joblib>=0.14.0
```
Install the packages with:

```bash
pip install -r requirements.txt
```

### Dataset Setup
The datasets used in this project are not included in this repository due to their size. You can download them from the following links:

**PlantVillage**: https://www.kaggle.com/datasets/tushar5harma/plant-village-dataset-updated
**PlantDoc**: https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset
After downloading and extracting, you should organize them in a dataset/ folder at the root of this project, following this structure:
```
Root/
└── dataset/
    ├── PlantVillage/
    │   ├── train/
    │   │   ├── Disease1/
    │   │   └── ...
    │   ├── val/
    │   └── test/
    └── PlantDoc/
        ├── train/  # The PlantDoc dataset from Kaggle includes train/val/test folders.
        ├── val/    # For this project, PlantDoc is primarily used for evaluating performance
        └── test/   # on field images, typically using its test set.
```
The PlantDoc dataset from Kaggle includes train, validation, and test splits. This project primarily utilizes the PlantDoc dataset to evaluate model performance on challenging field images.

### Models

This project requires models to be trained from scratch using the provided scripts. Trained model files are not included in the repository. When you run the training scripts, the resulting model files (e.g., .pkl, .h5, etc.) will be saved in the corresponding subdirectories under the models/ folder, as outlined in the Project Structure section.

## Usage

The `scripts/architecture_framework_scripts/` directory contains subdirectories for each implemented architecture (e.g., `convnext/`, `glcm_knn/`, `resnet50/`, etc.). Within each architecture's folder, you will find scripts for training, inference, and timing.

**General Script Types (examples from ConvNext, similar scripts exist for other architectures):**

* **Base Model Training:** `*_training.py` (e.g., `convnext_training.py`)
    * Trains the standard (base) version of the deep learning model.
* **Augmented Model Training:** `*_augmented_training.py` (e.g., `convnext_augmented_training.py`)
    * Trains the model using GLCM features as augmented input.
* **Stacking Meta-Learner Training:** `*_meta_learner.py` (e.g., `convnext_meta_learner.py`)
    * Trains the logistic regression model for the stacking approach, using outputs from base models.
* **Inference Scripts:** Various `*_inference.py` scripts (e.g., `convnext_inference.py`, `convnext_hierarchical_inference.py`, `convnext_augmented_inference.py`, `convnext_stacking_inference.py`)
    * Run these scripts to perform inference and evaluate the corresponding trained model/approach.
* **Time Measurement Scripts:** Found in the `time_scripts/` subfolder within each architecture's script directory (e.g., `scripts/architecture_framework_scripts/convnext/time_scripts/augmented_time.py`).
    * These scripts (`augmented_time.py`, `base_time.py`, `hierarchical_time.py`, `stacking_time.py`) measure the running time (without data loading overhead) for each respective approach using the specific architecture.

**Example Workflow (Conceptual):**

1.  **Navigate to the script directory for your chosen architecture:**
    ```bash
    cd scripts/architecture_framework_scripts/convnext/
    ```
2.  **Train a model (e.g., base ConvNext model):**
    ```bash
    python convnext_training.py # (You may need to pass arguments for dataset paths, epochs, etc.)
    ```
    *(Refer to the script's internal documentation or use `python <script_name>.py --help` if argument parsing is implemented).*
3.  **Run inference with the trained model:**
    ```bash
    python convnext_inference.py # (Arguments for model path, dataset path, etc.)
    ```
4.  **To measure execution time for an approach (e.g., augmented ConvNext approach):**
    ```bash
    cd time_scripts/
    python augmented_time.py # (Likely requires configuration or arguments pointing to the ConvNext augmented model components)
    ```

**Note:** Specific command-line arguments and configurations will depend on how each script is implemented. Please refer to the source code of individual scripts for detailed usage instructions. The `glcm+knn.py` script is located within `scripts/architecture_framework_scripts/glcm_knn/`.

## Results

The `results/` folder will store the outputs from the inference scripts. For each model and approach, you can typically find:
-   `.txt` files containing detailed classification metrics.
-   `.csv` files summarizing key metrics.
-   `.jpg` files showing the confusion matrix for the classification.

## Project Goals

The focus of this research is on addressing the degradation in accuracy observed when applying current plant disease detection models to raw field images. By developing and comparing various multi-stage hybrid classification frameworks, this project aims to:

-   Improve classification accuracy specifically on challenging field environment images.
-   Analyze the performance of different hybrid approaches (Sequential/Hierarchical, Augmented, Stacking) across multiple deep learning architectures.
-   Evaluate the contribution of traditional image features (GLCM) in hybrid frameworks.
-   Provide a flexible framework for experimenting with combinations of models.
-   Investigate training and inference times for the proposed frameworks.

## License

This project is licensed under the **GNU GPLv3 License**. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you have ideas for improvements or additional features, please open an issue or submit a pull request.
When using the code from this project, please ensure proper attribution by citing this work. If you intend to use this project or its derivatives for commercial purposes that may gain profit, please contact the maintainers to discuss potential arrangements.

## Acknowledgements

-   Thanks to the contributors of the various libraries and datasets used in this project.
-   Special thanks to the research community for inspiring hybrid approaches in plant disease detection.
