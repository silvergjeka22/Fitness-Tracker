# Fitness Tracker – Activity Recognition Using Sensor Data

## Introduction

**Fitness Tracker** is a project that uses machine learning to recognize different exercises from sensor data. It uses data from wearable devices, like accelerometers and gyroscopes, collected during gym workouts. The goal is to build models that can classify exercises based on the sensor readings.

This project demonstrates the full workflow: from data preprocessing and feature extraction to model training, evaluation, and performance comparison.

## Sensors Used

- **Accelerometer**: Measures body movement across the X, Y, and Z axes.
- **Gyroscope**: Captures rotational motion and orientation along three axes.

These sensors work together to capture a detailed picture of body motion during exercise.

## Dataset

The dataset used is publicly available on Kaggle and includes labeled sensor data from multiple gym exercises. It contains:

- 3-axis accelerometer data
- 3-axis gyroscope data
- Labels for different exercises
- Participant and intensity information

**Dataset Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/krishujeniya/fitness-tracker-accelerometer-and-gyroscope-data)

## Project Workflow

### 1. Outlier Detection

- DBScan clustering is used to detect outliers in the sensor data.
- Detected outliers are replaced using interpolation for data smoothing.

### 2. Feature Engineering

- Using Low pass filter to smooth the raw sensor signals.

### 3. Feature Extraction & Clustering

- Extracted statistical features such as mean, standard deviation, magnitude, etc.
- Applied Principal Component Analysis (PCA) to reduce dimensionality.
- Performed K-means clustering for visual inspection of patterns.

### 3. Models

- Trained and compared multiple machine learning models:
  - Random Forest
  - Decision Tree
  - K-Nearest Neighbors
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (Linear)

## How to Run the Project

1. **Clone the repository**.

2. **Set up the environment**:
   If using Conda:

   ```bash
   conda create -n fitness-env python=3.10
   conda activate fitness-env
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebook or script**:
   ```bash
   jupyter notebook
   ```
