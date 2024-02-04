

# Adult Income Prediction
# Overview
This project focuses on predicting adult income levels using machine learning algorithms. The dataset used for this task contains various demographic and socioeconomic attributes of individuals, such as age, education, occupation, and marital status, along with their corresponding income levels.

The primary objective is to build predictive models that can classify individuals into different income categories based on their attributes. This classification task is crucial for understanding income distribution patterns and identifying factors that influence income levels.

# Algorithms Used
## 1. K-Nearest Neighbors (KNN)
KNN is a simple and intuitive classification algorithm that classifies data points based on the majority class of their nearest neighbors in the feature space. It does not make any assumptions about the underlying data distribution and can handle both numerical and categorical data.

## 2. Weighted K-Nearest Neighbors (W_KNN)
W_KNN is an extension of the KNN algorithm that assigns weights to the nearest neighbors based on their distance from the query point. This modification gives more influence to closer neighbors, potentially improving the accuracy of the classification.

## 3. Naive Bayes
Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes that features are conditionally independent given the class label, making it computationally efficient and easy to implement. Naive Bayes is particularly effective for text classification tasks but can also be applied to numeric and categorical data.

# Implementation
The project involves the following steps:

Data Preprocessing: Cleaning the dataset, handling missing values, encoding categorical variables, and scaling numerical features.
Model Training: Splitting the dataset into training and testing sets, training the KNN, W_KNN, and Naive Bayes models on the training data.
Model Evaluation: Evaluating the performance of each model using accuracy metrics and visualizations.
Conclusion: Drawing insights from the model results and discussing the strengths and limitations of each algorithm.

## Results
After training and evaluating the models, the following accuracies were obtained:

KNN Accuracy: 78.35%
Weighted KNN Accuracy: 77.76%
Naive Bayes Accuracy: 79.91%

## Conclusion
In conclusion, this project demonstrates the application of machine learning algorithms for predicting adult income levels. While each algorithm has its strengths and weaknesses, all three models provide valuable insights into income classification based on demographic and socioeconomic attributes.
