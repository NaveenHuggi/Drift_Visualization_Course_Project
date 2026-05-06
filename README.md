# Deep Learning Drift Visualization

This repository contains a comprehensive Jupyter Notebook demonstrating how to detect and visualize model performance issues (such as data drift) using Class Activation Mapping (CAM) techniques.

## Project Overview

When deploying machine learning models, especially deep learning models, they may encounter data that differs from their training distribution (data drift). This notebook provides a complete pipeline to:
1. **Train** a baseline Convolutional Neural Network (`SimpleCNN`) on the **MNIST** dataset.
2. **Evaluate** the model on a different, but somewhat related, distribution (**SVHN** - Street View House Numbers dataset) to simulate data drift or out-of-distribution (OOD) data.
3. **Visualize** the model's focus and attention using 8 different Grad-CAM algorithms to understand *why* the model makes specific predictions and how data drift affects its reasoning.

## Key Components

- **Data Pipeline:**
  - Loads the MNIST dataset for training.
  - Applies transformations (resizing to 32x32, converting grayscale to 3-channel RGB) to match SVHN requirements.
  - Loads the SVHN dataset for testing and drift evaluation.
- **Model Architecture:**
  - `SimpleCNN`: A custom CNN architecture consisting of two convolutional layers with ReLU activation and Max-Pooling, followed by two fully connected layers.
- **Training & Evaluation:**
  - Implements a standard PyTorch training loop.
  - Uses CrossEntropyLoss and the Adam optimizer.
  - Visualizes Loss and Accuracy metrics using Matplotlib.
- **Interpretability & Visualization:**
  - Integrates the `grad-cam` library to generate heatmaps.
  - Compares 8 state-of-the-art CAM algorithms:
    1. GradCAM
    2. HiResCAM
    3. ScoreCAM
    4. GradCAM++
    5. AblationCAM
    6. XGradCAM
    7. EigenCAM
    8. FullGrad
  - Outputs a 3x3 grid visualizing how each algorithm interprets the model's focus on the input image.

## Requirements

To run this notebook, you will need the following Python libraries:
- `torch`
- `torchvision`
- `matplotlib`
- `numpy`
- `grad-cam`

Install them using:
```bash
pip install torch torchvision matplotlib numpy grad-cam
```

## Usage

1. Clone this repository.
2. Open `DeepLearning_Drift_Visualization.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run the cells sequentially to install dependencies, load data, train the model, and generate the visualizations.
