# Import necessary libraries for data handling, visualization, and machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For improved visualization aesthetics

# Read the dataset from a specified path into a pandas DataFrame
ds = pd.read_csv(r"C:\Users\prasu\DS2\git\classification\4. Naive bayes\Social_Network_Ads.csv")

# Extract the relevant features and the target variable from the dataset
X = ds.iloc[:, 2:4].values  # Typically columns 2 and 3 are features such as Age and Estimated Salary
y = ds.iloc[:, -1].values   # The target variable, typically in the last column, indicating the class (e.g., Purchased)

# Split the data into training and testing sets to evaluate the model's performance
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # 20% of data is used for testing

# Feature scaling to normalize data and improve the performance of the algorithm
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test_scaler = scaler.transform(X_test)        # Transform the test data using the same scale

# Initialize and train a Gaussian Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
algorithm = GaussianNB()
algorithm.fit(X_train_scaler, y_train)  # Train the model on the scaled training data

# Make predictions on the scaled test data
y_pred = algorithm.predict(X_test_scaler)

# Evaluate the accuracy of the classifier
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)  # Compute the accuracy by comparing the predicted and actual values

# Calculate and print the bias (training accuracy) and variance (test accuracy)
bias = algorithm.score(X_train_scaler, y_train)  # Training score reflects the model's bias
variance = algorithm.score(X_test_scaler, y_test)  # Test score reflects the model's variance

# Generate a confusion matrix to evaluate the detailed performance of the classifier
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)  # This matrix helps visualize true vs. predicted classifications

# Output the results to understand model performance
print("Accuracy:", accuracy)
print("Bias (Training Score):", bias)
print("Variance (Test Score):", variance)
print("Confusion Matrix:\n", cm)
