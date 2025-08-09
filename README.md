# Handwriting Digit Detection Project

This project focuses on recognizing handwritten digits (0–9) from the **MNIST dataset** using a **Convolutional Neural Network (CNN)**. The model is trained to classify grayscale digit images into their respective categories with high accuracy.

---

## Project Structure
1. **Data Loading**
   - MNIST dataset is loaded from IDX files using a custom `load_mnist` function.
   - The dataset contains grayscale images of size **28×28 pixels** and their corresponding digit labels.

2. **Data Preprocessing**
   - Images are normalized to values between 0 and 1 for better training performance.
   - The dataset is split into training and validation sets using `train_test_split`.

3. **Model Building**
   - A **Sequential CNN model** is created using TensorFlow Keras.
   - Layers include:
     - **Conv2D** layers for feature extraction.
     - **MaxPooling2D** layers for dimensionality reduction.
     - **Flatten** and **Dense** layers for classification.

4. **Training & Evaluation**
   - Model is compiled with **Adam optimizer** and `sparse_categorical_crossentropy` loss.
   - Training is performed for **5 epochs** with validation accuracy tracking.
   - Final validation accuracy achieved: **~98.94%**.

5. **Prediction on Custom Images**
   - The trained model is saved as `mnist_cnn_model.h5`.
   - Custom images are preprocessed (grayscale conversion, resizing to 28×28, color inversion, normalization) before prediction.
   - Predictions are visualized alongside the input image.

---

## Requirements
Install dependencies using:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

## Getting Started
1. Clone or download the project files.
2. Place the MNIST dataset files (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`) in the working directory.
3. Run the Jupyter Notebook or Python script to train the model.
4. Use the provided functions to predict digits from custom images.

---

## Dataset
- **Source**: MNIST handwritten digits dataset.
- **Format**: IDX files containing image and label data.
- **Image size**: 28×28 pixels, grayscale.

---

## Model Architecture
1. **Conv2D** → **MaxPooling2D**
2. **Conv2D** → **MaxPooling2D**
3. **Conv2D** → **Flatten**
4. **Dense** → **Dense (Softmax)**

---

## How to Use
1. Train the model by running the training script or notebook cells sequentially.
2. Save the trained model for future predictions.
3. Use the `predict_digit` function with the path to your image to get predictions.

---

## Results
- **Validation Accuracy**: ~98.94%
- **Validation Loss**: ~0.0327
- Model effectively classifies handwritten digits with high accuracy.
- Future improvements can include:
  - Data augmentation for more robust training.
  - Hyperparameter tuning for better performance.
