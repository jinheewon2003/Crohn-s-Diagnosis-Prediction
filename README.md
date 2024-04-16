# Random Forest Classifier for Predicting CD Diagnosis

This repository contains code for training and evaluating a Random Forest classifier to predict Crohn's Disease (CD) diagnosis using clinical data. 

## Introduction

Crohn's Disease (CD) is a chronic inflammatory condition of the gastrointestinal tract. Early diagnosis and treatment are crucial for managing CD effectively. Machine learning models, such as Random Forest classifiers, can assist in predicting CD diagnosis based on clinical data.

## Dataset

The dataset used in this project is sourced from a study published in PubMed Central (PMC). The original dataset can be found at: [PMC article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4059512/). It contains anonymized clinical data of patients, including demographic information, medical history, and diagnostic features.

## Code Overview

The Python code in this repository performs the following tasks:

1. Reads the dataset from an Excel file.
2. Preprocesses the data by filtering out irrelevant features and converting categorical variables into numerical form.
3. Imputes missing values using the mean strategy.
4. Splits the dataset into training and testing sets.
5. Normalizes the input features to a range of 0 to 1 using Min-Max scaling.
6. Trains a Random Forest classifier on the training data.
7. Evaluates the classifier's performance on the test data, calculating accuracy.
8. Prints the predicted and true values side by side, along with the test accuracy.

## Usage

To run the code:

1. Ensure you have Python installed on your system.
2. Clone this repository to your local machine.
3. Install the required Python packages: `pip install pandas scikit-learn tensorflow numpy`.
4. Download the dataset from the provided link and place it in the repository directory.
5. Run the Python script `random_forest_classifier.py`.

## Results

The trained Random Forest classifier achieves an accuracy of 98.97% on the test data. The predicted and true values are printed side by side for comparison. 
