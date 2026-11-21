# Experiment 5 : Multilayer Perceptron (MLP) Neural Network

## Aim

To implement a Multilayer Perceptron (MLP) neural network for
handwritten digit classification using Python and evaluate its
performance on the test dataset.

> **Note:** The source page contains an inconsistent aim statement
> mentioning K-Means clustering. The experiment title and algorithm
> steps describe an MLP neural network, so this README follows the MLP
> experiment.

## Overview

A **Multilayer Perceptron (MLP)** is a feed-forward artificial neural
network consisting of an input layer, one or more hidden layers, and an
output layer. It learns patterns in data by adjusting weights and biases
during training using backpropagation.

In this experiment, the MLP model is trained on the **Digits dataset**
available in scikit-learn. Each sample is an 8 × 8 image of a
handwritten digit (0--9). The pixel values are used as input features,
and the model predicts the corresponding digit.

## Algorithm

1.  **Start.**
2.  Load the Digits dataset and separate the input features and target
    labels.
3.  Split the dataset into training and testing sets.
4.  Standardize the input features.
5.  Create an MLP classifier with the specified hidden-layer structure.
6.  Train the model using the training data.
7.  Predict the labels for the test data.
8.  Evaluate the model using test accuracy and a classification report.
9.  Plot the training loss curve.
10. **Stop.**
ry slightly with changes to the train-test split,
    random seed, or model parameters.


## Program
import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score,classification_report

# Load digits dataset

digits = load_digits()

X = digits.data

y = digits.target

# Split and scale

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)

X_test_s = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(64,), activation='relu',

solver='adam', max_iter=300, random_state=42)

mlp.fit(X_train_s, y_train)

# Evaluate

y_pred=mlp.predict(X_test_s)

acc = accuracy_score(y_test, y_pred)

print("MLP Test Accuracy:", acc)

print(classification_report(y_test, y_pred))

# Plot loss curve

plt.figure(figsize=(6,4))

plt.plot(mlp.loss_curve_)

plt.title('MLP Loss Curve')

plt.xlabel('Iterations')

plt.ylabel('Loss')

plt.show()




## output

MLP Test Accuracy: 0.98
              precision    recall  f1-score   support

           0       0.98      0.98      0.98        45
           1       0.96      0.98      0.97        46
           2       0.96      1.00      0.98        44
           3       1.00      1.00      1.00        46
           4       0.98      0.98      0.98        45
           5       1.00      0.96      0.98        46
           6       1.00      1.00      1.00        45
           7       0.98      1.00      0.99        45
           8       1.00      0.91      0.95        43
           9       0.96      1.00      0.98        45

    accuracy                           0.98       450
   macro avg       0.98      0.98      0.98       450
weighted avg       0.98      0.98      0.98       450

<img width="667" height="481" alt="image" src="https://github.com/user-attachments/assets/b8edb382-2016-4f2f-9049-98f62cdd887f" />


## Result

