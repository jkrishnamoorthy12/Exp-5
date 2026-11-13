# Experiment 8: Multilayer Perceptron (MLP) Neural Network

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

## Requirements

-   Python 3
-   NumPy
-   Matplotlib
-   scikit-learn

Install the required packages:

``` bash
pip install numpy matplotlib scikit-learn
```

## Python Implementation

Save the following program as `mlp_digits.py`.

``` python
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


# Step 1: Load the Digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 3: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Create the MLP neural network
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42,
    early_stopping=True
)

# Step 5: Train the model
mlp.fit(X_train_scaled, y_train)

# Step 6: Predict the test data
y_pred = mlp.predict(X_test_scaled)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Plot the training loss curve
plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_)
plt.title("MLP Neural Network - Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
```

## How to Run

1.  Install Python 3 and the required libraries.

2.  Save the program as `mlp_digits.py`.

3.  Open a terminal or command prompt in the folder containing the
    program.

4.  Run the following command:

    ``` bash
    python mlp_digits.py
    ```

5.  Observe the test accuracy, classification report, and training loss
    curve.

## Expected Output

The program displays the test accuracy and a classification report
containing precision, recall, F1-score, and support for each digit class
(0--9).

Example output format:

``` text
Test Accuracy: <calculated accuracy> %

Classification Report:
              precision    recall  f1-score   support
           0       ...
           1       ...
           2       ...
           ...
           9       ...
```

The exact values may vary depending on the software environment and
model settings. A training loss curve is also displayed.

## Result

The Multilayer Perceptron neural network is implemented using the
scikit-learn Digits dataset. The model is trained on standardized
features and evaluated using test accuracy and a classification report.
The training loss curve is plotted to observe how the model's loss
changes during training.

## Notes

-   The Digits dataset contains 8 × 8 images represented by 64 numerical
    pixel features.
-   `hidden_layer_sizes=(100, 50)` specifies two hidden layers
    containing 100 and 50 neurons.
-   `StandardScaler` is fitted only on the training data to prevent data
    leakage.
-   The loss curve shows the training loss over iterations; a decreasing
    curve generally indicates that the model is learning.
-   Results may vary slightly with changes to the train-test split,
    random seed, or model parameters.
