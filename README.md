# Retinal Image Processing with GANs

## Overview
This repository implements a deep learning framework for processing retinal images using Generative Adversarial Networks (GANs) and classifiers. The project aims to classify images into Diabetic Macular Edema (DME) and Drusen, while enhancing image resolution.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Training the Model](#training-the-model)
- [Metrics Tracking](#metrics-tracking)
- [Results Visualization](#results-visualization)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have the following installed:
* Python 3.7 or higher
* PyTorch
* torchvision
* NumPy
* Matplotlib
* scikit-learn
* tqdm
* PIL (Pillow)

Install the required packages using pip:
```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm pillow
```

## Installation

1. Clone this repository to your local machine:
```bash
git clone https://github.com/yourusername/retinal-image-processing.git
cd retinal-image-processing
```

## Dataset Preparation

### Dataset Structure
Ensure your dataset is organized as follows:
```
dataset/
├── train/
│   ├── DME/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── DRUSEN/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── test/
    ├── DME/
    └── DRUSEN/
```

### Configuration
Update the dataset path in the `RetinalDataset` class within the `main()` function to point to your dataset directory.

## Usage

To train the models, run the main script:
```bash
python main.py
```

For specific configurations, you can pass additional arguments:
```bash
python main.py --batch-size 32 --epochs 100 --learning-rate 0.001
```

## Code Structure
```
.
├── main.py
├── models/
│   ├── generator.py
│   ├── discriminator.py
│   └── classifier.py
├── utils/
│   ├── data_loader.py
│   ├── transforms.py
│   └── metrics.py
└── config/
    └── config.yaml
```

## Training the Model

The training process consists of two phases:
1. GAN training for image enhancement
2. Classifier training for disease detection

## Metrics Tracking

The following metrics are tracked during training:
* Generator Loss
* Discriminator Loss
* Classification Accuracy
* Precision
* Recall
* F1 Score

## Results Visualization

Results and metrics are automatically saved in the `results` directory:
```
results/
├── figures/
│   ├── loss_curves.png
│   └── confusion_matrix.png
└── checkpoints/
    ├── generator_best.pth
    └── classifier_best.pth
```


