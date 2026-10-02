# # Comparing Linear Regression models

#import modules and data
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('housing.csv', sep = ',', header =None)
df.head()

df.shape
df.columns = ['CRIM','ZN','INDUS', 'CHAS' , 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT' , 'MEDV' ]
print(df.head())
data = df.values

print(df.isnull().sum())

# using the correlation matrix you can choose the variables that highly correlated with our target variable
sns.set(rc = {'figure.figsize':(15,8)})
sns.heatmap(df.corr(), annot = True, fmt = ".1f")
plt.show()

#column_name=['INDUS','RM','TAX','PTRATIO','LSTAT']
#X, y = df[column_name], df['MEDV'] 
x, y = data[:,:-1], data[:,-1] 

# ## Prediction with Linear Regression 

from sklearn.model_selection import train_test_split #Split arrays or matrices into
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from collections import Counter

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
model1 = LinearRegression()
model1.fit(X_train, y_train)
y_1 = model1.predict(X_test)
print("R^2: ", r2_score(y_test,y_1))
print("MSE:", mean_squared_error(y_1,y_test))
#print(y_1)

from sklearn.model_selection import cross_val_score
cv_1= cross_val_score(model1,x,y, cv = 10)
print(cv_1)
cv_1 = np.absolute(cv_1)
print('Mean MAE: %.3f (%.3f)' % (np.mean(cv_1), np.std(cv_1)))

# ## Prediction with Ridge Regression

from sklearn.linear_model import Ridge
model2 = Ridge(alpha=0.08)
model2.fit(X_train, y_train)
y_2 = model2.predict(X_test)
print("R^2: ", r2_score(y_test,y_2))
print("MSE:", mean_squared_error(y_2,y_test))
#print(y_2)

# First Way
cv_2= cross_val_score(model2,x,y, cv = 10)
cv_2 = np.absolute(cv_2)
print(cv_2)
print('Mean MAE: %.3f (%.3f)' % (np.mean(cv_2), np.std(cv_2)))


#Second Way
from sklearn.model_selection import KFold
cv = KFold(n_splits=10, random_state=1)
# evaluate model
scores = cross_val_score(model2, x, y, cv=cv)
print(scores)
# force scores to be positive
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# ## Prediction with Lasso Regression

from sklearn.linear_model import Lasso
model3 = Lasso(alpha=1)
model3.fit(X_train, y_train)
y_3 = model3.predict(X_test)
print("R^2: ", r2_score(y_test,y_3))
print("MSE:", mean_squared_error(y_3,y_test))
#print(y_3)

cv_3= cross_val_score(model3,X,y, cv = 10)
cv_3 = np.absolute(cv_3)
print('Mean MAE: %.3f (%.3f)' % (np.mean(cv_3), np.std(cv_3)))

# ### Choosing lambda

grid = dict()
grid['alpha'] = np.arange(0, 1, 0.01)
print(grid)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
model = Ridge()
# define model evaluation method
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# define grid
grid = dict()
grid['alpha'] = np.arange(0, 1, 0.01)
# define search
search = GridSearchCV(model, grid, cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

# Exercise

### Generate data from a linear model with controllable noise and number of features.

Use the Diabetes dataset from scikit-learn:

from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True)

### Task 1 — Data preparation
- Load the dataset and split into training and test sets (train_test_split, 80/20 split).
- Standardize features (mean=0, variance=1) using StandardScaler.

### Task 2 — Baseline model
- Fit an Ordinary Least Squares (LinearRegression) model.
- Record training and test RMSE and R^2.
- Save coefficients for comparison.

### Task 3 — Ridge regression
- Train Ridge regression with a grid of alpha values (e.g., [0.01, 0.1, 1, 10, 100]).
- Plot training and test RMSE vs log(alpha) to visualize the bias–variance tradeoff.
- Compare Ridge coefficients to OLS.
- Plot Ridge coefficient paths as alpha varies (use sklearn.linear_model.Ridge with alphas grid).

### Task 4 — Lasso regression
- Train Lasso regression with the same grid of alpha values.
- Plot training and test RMSE vs log(alpha).
- Plot the number of non-zero coefficients vs alpha.
- Plot Lasso coefficient paths across a fine grid of alphas.

### Task 5 — Cross-validation
- Use GridSearchCV with 10-fold CV to select the best alpha for Ridge and Lasso.
- Report selected alpha values and corresponding RMSE on the test set.

### Task 6 — Bias–Variance tradeoff (extended)
- Perform repeated train/test splits (e.g., 30 repetitions).
- For each alpha, compute average training RMSE, test RMSE, and their variances.
- Plot curves showing how training error, test error, and variance change with log(alpha).
- Discuss where bias dominates and where variance dominates.

