# MNIST Dataset Analysis

This project performs an exploratory data analysis (EDA) on the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset using TensorFlow and related libraries such as `tensorflow-datasets`, `matplotlib`, `seaborn`, and `pandas`.

## 🔍 Overview

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9), divided into:
- **Training set**: 60,000 images
- **Test set**: 10,000 images

Each image is 28x28 pixels in size.

This script analyzes both the training and test datasets by:
- Plotting class distributions
- Displaying example images
- Checking if the dataset is balanced

## 🧰 Requirements

To run this code, you need the following libraries installed:

```bash
pip install tensorflow tensorflow-datasets matplotlib seaborn pandas numpy
```

## 📁 Files

- `dataset_analysis.py`: Main script to load and analyze the MNIST dataset.
- `README.md`: This file.

## 📈 Visualizations

### Class Distribution Bar Chart
A bar plot showing how many samples are present for each digit class.
![](https://github.com/ectorr01/MNIST-Dataset-Analysis/blob/main/Visualizations/Figure_1.png)


### Sample Image Grid
A 3x3 grid displaying sample images from the dataset along with their corresponding labels.
![](https://github.com/ectorr01/MNIST-Dataset-Analysis/blob/main/Visualizations/Figure_2.png)

### Class Distribution Heatmap
A heatmap summarizing the count of images per class in a compact format.
![](https://github.com/ectorr01/MNIST-Dataset-Analysis/blob/main/Visualizations/Figure_3.png)

## 🚀 Usage

To run the analysis:

```bash
python dataset_analysis.py
```

The script will display plots and print statistics about the dataset.

## 📦 Next Steps

In future commits, we will add:
- A deep learning model built with Keras
- Model training and evaluation
- Model inference and saving/loading functionality

Stay tuned!
```
