#!/usr/bin/env python
# coding: utf-8

# # Laboratory work 2. Linear Regression

# # Task 1. Univariate 

# Read the file ('Student_Marks.csv'). 
# 
# - Import LinearRegression from scikit-learn library. Create a linear regression model, fit it to the training set and evaluate on the test set. 
# 
# - Compute R2 using model.score. 
# 
# - Compute Mean Squres Error using mean_squared_error() function.
# 
# - Add polynomial features (degree 2,3,4).
# 
# - Evaluate if linear regression still performs well or if overfitting occurs.

# # Task 2. Multivariate

# ### Problem Statement
# Read the file ('house_price_regression_dataset.csv'). 
# 
# You are tasked with building a linear regression model to predict house prices using a real-world dataset. The dataset contains both numerical and categorical features, some with missing values and strong correlations. Your goal is not just to fit a model but to overcome challenges like multicollinearity, feature scaling, and overfitting.

# ### Data Preprocessing
# - Handle missing values (decide: drop, mean/median imputation, or domain-specific).
# 
# - Convert categorical variables into numerical form.
# 
# - Normalize/standardize numerical features where necessary.

# ### Exploratory Data Analysis (EDA)
# - Plot correlations between numerical features and target. 
# 
# - Detect multicollinearity (hint: Variance Inflation Factor).
# 
# - Select the top 15 most relevant features using correlation, domain knowledge, or statistical tests.

# ### Model Building
# - Implement Ordinary Least Squares Linear Regression (without sklearn, write code from scratch to compute β = (XᵀX)⁻¹Xᵀy).
# 
# - Compare your custom implementation with sklearn.linear_model.LinearRegression.

# ### Model Evaluation
# - Split data into train/validation/test sets.
# 
# - Evaluate models using RMSE, MAE, and R².
