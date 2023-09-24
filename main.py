import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("wine.csv", sep=";")

print(df.head(100))

print(df.shape)

print(df.info())

print(df.describe())
print(df.isna().sum())

# fig = px.histogram(df, x='1')
# fig.show()


# Preprocessing

# Splitting the data into training and test sets
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
print(len(train_set))
print(len(test_set))

wine = train_set.copy()
print(wine.head(100))
wine = train_set.drop("quality", axis=1)
wine_labels = train_set["quality"].copy()
print(wine_labels.head(100))
print(wine)

# No need to use imputer or something for missing values because there are no missing values in the dataset
# As long as all features are numerical, the best practice is to scale all of them to the range [0, 1]

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])
wine_tr = num_pipeline.fit_transform(wine)
num_attribs = list(wine)
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs)
])
wine_prepared = full_pipeline.fit_transform(wine)
print(wine_prepared[0:100])

# Training Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(wine_prepared,wine_labels)
wine_pred = lin_reg.predict(wine_prepared)
lin_mse = mean_squared_error(wine_labels, wine_pred)
lin_rmse = np.sqrt(lin_mse)
print(f"Linear Regression RMSE for Train Set: {lin_rmse}")

# Testing on Test Set
wine_test = test_set.drop("quality", axis=1)
wine_test_labels = test_set["quality"].copy()
wine_test_prepared = full_pipeline.transform(wine_test)
wine_pred = lin_reg.predict(wine_test_prepared)
lin_mse = mean_squared_error(wine_test_labels, wine_pred)
lin_rmse = np.sqrt(lin_mse)
print(f"Linear Regression RMSE for Test Set: {lin_rmse}")


# Logistic Regression
lg_reg = LogisticRegression()
lg_reg.fit(wine_prepared, wine_labels)
lg_pred = lg_reg.predict(wine_prepared)
lg_mse = mean_squared_error(wine_labels, lg_pred)
lg_rmse = np.sqrt(lg_mse)
print(f"Logistic Regression RMSE for Train Set: {lg_rmse}")

# Testing on Test Set
wine_test_prepared = full_pipeline.transform(wine_test)
lg_pred = lg_reg.predict(wine_test_prepared)
lg_mse = mean_squared_error(wine_test_labels, lg_pred)
lg_rmse = np.sqrt(lg_mse)
print(f"Logistic Regression RMSE for Test Set: {lg_rmse}")


# Try another Linear Model like DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(wine_prepared, wine_labels)
tree_pred = tree_reg.predict(wine_prepared)
tree_mse = mean_squared_error(wine_labels, tree_pred)
tree_rmse = np.sqrt(tree_mse)
print(f"Decision Tree Regression RMSE for Train Set: {tree_rmse}")

# Testing on Test Set
wine_test_prepared = full_pipeline.transform(wine_test)
tree_pred = tree_reg.predict(wine_test_prepared)
tree_mse = mean_squared_error(wine_test_labels, tree_pred)
tree_rmse = np.sqrt(tree_mse)
print(f"Decision Tree Regression RMSE for Test Set: {tree_rmse}")







# Unfortunately, it takes a lot of time, so commented out
# X = df.iloc[:, 0:-1]
# y = df.iloc[:, -1]
# loo = LeaveOneOut()
# logreg = LogisticRegression()
# score = cross_val_score(logreg, X, y, cv=loo)
# print("Cross Validation Scores are {}".format(score))
# print("Average Cross Validation score :{}".format(score.mean()))





"""
For this exercise, I watched the video lectures that you provided in the WyoCourse once again to learn more about the concepts of regression and classification
I also read some of the materials on the web about how to code nicely in machine learning and python
But mostly I read two chapters of the book named "Hands-On Machine Learning with scikit-learn and tensorflow"
I reallu learned a lot by reading this book, especially the concepts in preprocessing section. That was so practical
"""

"""
RESULTS: 
Linear Regression RMSE for Train Set: 0.6512995910592837
Linear Regression RMSE for Test Set: 0.624519930798013
Logistic Regression RMSE for Train Set: 0.702948108884437
Logistic Regression RMSE for Test Set: 0.700446286306095
Decision Tree Regression RMSE for Train Set: 0.0
Decision Tree Regression RMSE for Test Set: 0.8139410298049853
"""

# It Seems like decision tree is more prone to overfitting, especially when trees are deep
