# Income Prediction: U.S. Census Data Modeling

###### This is a **Machine Learning project** focused on predicting individuals' income using the 1994 U.S. Census dataset. The project employs various supervised learning algorithms to predict whether an individual's income exceeds $50,000, based on personal and demographic features.

## Overview

The **Income Prediction** project aims to leverage machine learning techniques to predict whether an individual's income exceeds $50,000 based on publicly available census data. The dataset includes various features such as age, education, occupation, and marital status. By utilizing a series of supervised learning algorithms, the goal is to determine which model performs best at predicting income class and then optimize that model for better accuracy.

This README will guide you through the project features, installation instructions, and provide an overview of the methodologies and models used.

<details>
  <summary><b>Table of Contents</b></summary>
  <ol>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#Models">Models</a></li>
  </ol>
</details>

## Overview

The **Income Prediction** project uses data collected from the 1994 U.S. Census to build a classification model. The goal is to predict whether an individual earns more than $50,000 based on various demographic and employment features. By employing multiple supervised machine learning models, the project aims to find the best algorithm and optimize it for accuracy.

This project is relevant for non-profit organizations that wish to understand the income distribution of potential donors or individuals they may wish to reach out to. The ability to predict income levels could inform strategies for donation requests or outreach.

###### Built With

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue) ![Scikit-Learn](https://img.shields.io/badge/scikit-learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-777BB4?style=for-the-badge&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-2C6BB0?style=for-the-badge&logo=matplotlib&logoColor=white)

## Features

### 1. **Income Prediction Model**

- **Purpose**: Predict whether an individual earns more than $50,000 using demographic and employment data.
- **Approach**:
  - Train multiple supervised machine learning models to classify income as either `<=50K` or `>50K`.
  - Evaluate model performance using various metrics such as accuracy, precision, recall, and F1-score.

### 2. **Data Preprocessing**

- **Purpose**: Clean and prepare the data for modeling.
- **Tasks**:
  - Handling missing data and ill-formatted records.
  - Encoding categorical variables.
  - Normalizing or scaling numerical features if needed.

### 3. **Supervised Learning Algorithms**

- **Purpose**: Apply a range of algorithms to determine which model performs best.
- **Algorithms**:
  - Logistic Regression
  - Decision Trees
  - Random Forests
  - Support Vector Machines (SVM)
  - Gradient Boosting Machines (GBM)

### 4. **Model Evaluation and Optimization**

- **Purpose**: Select the best performing algorithm and optimize it.
- **Techniques**:
  - Cross-validation
  - Hyperparameter tuning (e.g., GridSearchCV, RandomizedSearchCV)
  - Evaluation metrics (accuracy, precision, recall, F1-score)

## Installation

### Prerequisites

Before running the **Income Prediction** project, ensure you have the following installed:

- **Python 3.x**
- **Required Libraries**

Some of the core libraries include:

- pandas: Data manipulation and cleaning.
- numpy: Numerical computations.
- scikit-learn: Machine learning models and utilities.
- matplotlib, seaborn: Data visualization.

## Models Used

This project employs a variety of machine learning models, including:

### 1. Logistic Regression

- **Type**: Linear model used for binary classification.
- **Use Case**: Predicting whether an individual's income is greater than $50,000.

### 2. Decision Trees

- **Type**: Non-linear model based on binary decision-making.
- **Use Case**: Easy to interpret and can capture complex relationships in data.

### 3. Random Forests

- **Type**: Ensemble model that combines multiple decision trees.
- **Use Case**: Robust model that generally performs well on structured data.

### 4. Support Vector Machines (SVM)

- **Type**: Classification model that works well for high-dimensional spaces.
- **Use Case**: Used when the data is not linearly separable.

### 5. Gradient Boosting Machines (GBM)

- **Type**: Ensemble method that builds models sequentially to reduce bias.
- **Use Case**: High-performing model, especially with unstructured data.

## Conclusion

The **Income Prediction** project leverages various machine learning models to predict whether an individual’s income exceeds $50,000 based on a set of demographic and employment features. By experimenting with different algorithms and optimizing the best model, the project aims to provide an accurate and reliable model for income classification.
