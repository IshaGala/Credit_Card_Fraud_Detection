# 🛡️ Credit Card Fraud Detection

This project is a machine learning implementation to detect fraudulent credit card transactions using Logistic Regression. The dataset is highly imbalanced, with the majority of transactions being legitimate and a very small proportion being fraudulent.

## 📁 Dataset

The dataset used in this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. It contains transactions made by European cardholders in September 2013.

- **Number of transactions**: 284,807  
- **Fraudulent transactions**: 492  
- **Features**: 30 (anonymized PCA components + Time and Amount)

## ⚙️ Features

- Data loading and basic exploration
- Handling data imbalance using **under-sampling**
- Model training with **Logistic Regression**
- Performance evaluation using:
  - Accuracy
  - Confusion Matrix
  - Classification Report
  - ROC-AUC Score

## 🧪 Techniques Used

- Data Cleaning
- EDA (Exploratory Data Analysis)
- Under-sampling for imbalanced data
- Logistic Regression
- Model Evaluation Metrics

## 📊 Results

- Achieved decent accuracy and ROC-AUC despite data imbalance
- Evaluated using training/test split with `stratify` to maintain class balance
- Outputs:
  - Accuracy on training and test datasets
  - Confusion Matrix
  - Classification Report
  - ROC-AUC Score

## 🚀 Getting Started

### Prerequisites

Ensure you have the following libraries installed:

```bash
pip install numpy pandas scikit-learn
