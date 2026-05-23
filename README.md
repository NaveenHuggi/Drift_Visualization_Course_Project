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
    1. **GradCAM**: Uses the gradients of the target concept flowing into the final convolutional layer to produce a coarse localization map highlighting important regions in the image.
    2. **HiResCAM**: Similar to GradCAM but calculates the weights differently, preserving higher resolution spatial details and preventing cross-talk between different features.
    3. **ScoreCAM**: A gradient-free visual explanation method that uses the increase in confidence score when passing a mask multiplied with the input image to determine weight, making it less noisy.
    4. **GradCAM++**: An extension of GradCAM that uses second-order gradients to provide better localization of objects, especially when there are multiple occurrences of the same class in an image.
    5. **AblationCAM**: A gradient-free method that systematically ablates (zeroes out) individual feature map channels to measure their impact on the final prediction, often resulting in cleaner heatmaps.
    6. **XGradCAM**: An improvement over GradCAM that seeks to achieve better theoretical grounding and faithfulness to the model by adjusting the gradient weights using normalized feature maps.
    7. **EigenCAM**: Computes the principal components (eigenvectors) of the feature maps, focusing on the most dominant patterns learned by the network without relying on class-specific gradients or backpropagation.
    8. **FullGrad**: Aggregates the gradients of the biases from all convolutional layers in addition to the input image gradients, capturing both local and global importance across the entire network.
  - Outputs a 3x3 grid visualizing how each algorithm interprets the model's focus on the input image.

## Example on Cancer Datasets

To demonstrate that the drift visualization and CAM techniques are widely applicable across different domains, here is an example of the methodology applied to a cancer-based dataset (e.g., Melanoma detection):

![Cancer Dataset Example](image.png)

This proves the utility of these interpretability methods beyond simple datasets (like MNIST/SVHN) and highlights their effectiveness on complex medical imaging tasks.

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
