# Twitter Spam Detection Project

A machine learning project for classifying tweets as spam or non-spam “ham” using text-based features and supervised learning.


## Project Overview

The objective of this project is to build, evaluate and tune a machine learning model that detects spam tweets based on text features. This project follows a structured Machine Learning workflow. This includes, from data preprocessing to model tuning and evaluation. 

The workflow covers: 

- Supervised Learning (classification) , 
- Text feature extraction (TF–IDF) , 
- Overfitting and generalization, 
- Model evaluation metrics (Precision, Recall, F1, ROC–AUC).

## Dataset
The project uses a Twitter spam dataset from Kaggle.

Columns used:
- `tweets` — the tweet text
- `class` — the label (`spam` or `ham`)

After preprocessing, a cleaned version of the data is saved for modeling.

## Project Structure

```
twitter_spam_detection/
├── README.md
├── main.py                        # Runs the full pipeline
├── 01_preprocessing.py            # Data cleaning and text normalization
├── 02_split_data.py               # Train / validation / test split + TF-IDF
├── 03_train_initial_model.py      # Baseline model training (e.g. Logistic Regression, Random Forest)
├── 04_evaluate_model.py           # Evaluation of initial models on validation set
├── 05_tune_model.py               # Logistic Regression tuning
├── 06_final_test.py               # Final evaluation of the selected / tuned model on test set
├── data/
│   ├── Twitter_Data.csv           # Original Kaggle dataset
│   ├── Twitter_Data_Clean.csv     # Cleaned dataset (from 01_preprocessing.py)
│   ├── split_data.joblib          # Saved splits and vectorizer (from 02_split_data.py)
│   ├── model_lr_tuned.joblib      # Final tuned Logistic Regression model
│   └── models_initial.joblib      # Initial baseline model
└── imagens/
    ├── confusion_matrix_initial.png
    ├── confusion_matrix_tuned.png
    └── roc_curve_lr.png
```


## Features

- **Text Preprocessing**: Cleaning tweets (URLs, mentions, punctuation)
- **TF-IDF Vectorization**: Converts text into numeric feature vectors
- **Supervised Learning Model**: Logistic Regression for binary classification
- **Model Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC
- **Hyperparameter Tuning**: Improves model performance
- **Visualization**: Saves ROC curve and confusion matrices


## Installation

```bash

cd twitter_spam_detection

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn joblib

```


## Usage

Run the scripts in order:


```bash
# Run the main.py

python3 main.py

```


## Machine Learning Concepts

| Concept                          | Implementation                                   |
| -------------------------------- | ------------------------------------------------ |
| **Supervised Learning**          | Logistic Regression model                        |
| **Data Preprocessing**           | Text cleaning, token normalization               |
| **Overfitting / Generalization** | Train/test split and cross-validation            |
| **Evaluation Metrics**           | Confusion matrix, Precision, Recall, F1, ROC-AUC |
| **Feature Extraction**           | TF–IDF vectorization of tweet text               |               |
| **Visualization**                | ROC curve, confusion matrix                      |


## Results

After running the full pipeline:
The baseline model achieves moderate performance using TF-IDF + Logistic Regression.
The tuned model improves F1-score and AUC after hyperparameter optimization.
All final plots and metrics are saved under the imagens/ directory.


### Visualizations
These plots are generated in 04_evaluate_model.py, it visualizes Random Forest feature importances and Logistic Regression coefficients, and compares confusion matrices between models.

Feature importance plots
Confusion metrics comparison
ROC curve

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib




