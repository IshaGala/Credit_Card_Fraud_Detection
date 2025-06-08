# Importing all dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Loading the dataset into a pandas dataframe
credit_card_data = pd.read_csv('creditcard.csv')

# First 5 rows of the dataset
print(credit_card_data.head())

# Last 5 rows of the dataset
print(credit_card_data.tail())

# Dataset information (like column data types, non-null counts)
print(credit_card_data.info())

# Checking the number of missing values in each column
print(credit_card_data.isnull().sum())

# Distribution of legit transactions and fraudulent transactions
print(credit_card_data['Class'].value_counts())

# This dataset is highly unbalanced
# 0 --> Normal Transaction
# 1 --> Fraudulent Transaction

# Separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Displaying the shapes of the two data sets
print("Legit Transactions Shape: ", legit.shape)
print("Fraudulent Transactions Shape: ", fraud.shape)

# Statistical measures for the Amount feature
print("Legit Amount Stats: ", legit.Amount.describe())
print("Fraud Amount Stats: ", fraud.Amount.describe())

# Compare the values for both transactions using groupby
print("Average Values for Legit and Fraudulent Transactions: ")
print(credit_card_data.groupby('Class').mean())

# Under-Sampling
# Build a sample dataset with similar distribution of normal and fraudulent transactions
# Number of fraudulent transactions = 492
legit_sample = legit.sample(n=492)

# Concatenating two dataframes
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Preview the new dataset
print(new_dataset.head())
print(new_dataset.tail())

# Check the distribution again
print("New Dataset Value Counts: ")
print(new_dataset['Class'].value_counts())
print("Average Values for Legit and Fraudulent in New Dataset: ")
print(new_dataset.groupby('Class').mean())

# Splitting the data into features (X) and target (Y)
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Display X and Y
print("Features (X): ")
print(X.head())
print("Target (Y): ")
print(Y.head())

# Splitting the data into training and testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Display the shapes of the split data
print("Shape of X:", X.shape)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# Model Training with Logistic Regression
model = LogisticRegression(max_iter=100)

# Training the logistic regression model
model.fit(X_train, Y_train)

# Model Evaluation
# Accuracy on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training Data: ', training_data_accuracy)

# Accuracy on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test Data: ', test_data_accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, X_test_prediction)
print("Confusion Matrix: ")
print(conf_matrix)

# Classification Report
print("Classification Report: ")
print(classification_report(Y_test, X_test_prediction))

# ROC-AUC Score
roc_auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])
print(f"ROC-AUC Score: {roc_auc}")
