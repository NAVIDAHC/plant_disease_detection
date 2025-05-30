# Plant Disease Detection: Hybrid Machine Learning & Deep Learning Approaches

This repository contains code for a hybrid approach to plant disease detection. The goal is to improve the accuracy of plant disease classification, especially in raw field images where current popular models often show degraded performance. The project combines traditional image processing (GLCM feature extraction), machine learning (KNN classifier), and deep learning methods in a multi-stage framework.

## Overview

The project is organized into three primary approaches:

1. **Sequential Approach:**
   - Uses **GLCM + KNN** for initial classification.
   - If an image is classified as diseased, a second-stage deep learning model refines the diagnosis.
2. **Augmented Approach:**
   - The output from the GLCM + KNN step serves as metadata for a second-stage classifier.
3. **Combination Approach:**
   - Combines GLCM, KNN, and a deep learning model into a single system for final classification.

This repository currently includes the **GLCM + KNN** implementation.

## Project Structure

```
Root/
├── dataset/
│   ├── PlantVillage/          # Laboratory images (train, val, test)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── PlantDoc/              # Field images (test only)
│       └── test/
├── models/                    # Saved trained models (.pkl)
├── results/                   # Evaluation results (CSV and TXT)
├── scripts/
│   └── glcm+knn.py           # GLCM + KNN implementation script
├── requirements.txt           # Required Python packages
└── README.md                  # This file
```

## Prerequisites

- **Python 3.6+** is required.
- Install the necessary packages using the `requirements.txt` file.

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

## Usage

1. **Dataset Setup:**  
   Organize your dataset with the following structure:

   ```
   Root/
   └── dataset/
       ├── PlantVillage/
       │   ├── train/
       │   ├── val/
       │   └── test/
       └── PlantDoc/
           └── test/
   ```

2. **Run the Code:**  
   Execute the GLCM + KNN script by running:

   ```bash
   python scripts/glcm+knn.py
   ```

   The script will:

   - Extract GLCM features from images.
   - Apply SMOTE oversampling to balance the training set.
   - Train a KNN classifier.
   - Evaluate the model on the test set.
   - Save evaluation results (both CSV and TXT formats) in the `results` folder.
   - Save the trained KNN model as a `.pkl` file in the `models` folder.

## Project Goals

The project's title is evolving to better reflect its goals. The focus is on addressing the degradation in accuracy observed when applying current models to raw field images. By using a combination of traditional feature extraction and advanced classification techniques, the project aims to:

- **Improve accuracy** on field environment images.
- **Reduce training time** by reusing pre-trained components.
- **Offer a flexible framework** to experiment with various combinations of models.

<!-- ## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have ideas for improvements or additional features.

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to the contributors of the various libraries and datasets used in this project.
- Special thanks to the research community for inspiring hybrid approaches in plant disease detection. -->
