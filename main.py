import numpy as np
import pandas as pd
import matplotlib as plt
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("/Users/alitorabi/Documents/Projects/pythonProject/Practical ML/warmup/wine.data")

print(df.head(100))

print(df.shape)

print(df.info())

print(df.describe())
print(df.isna().sum())

# fig = px.histogram(df, x='1')
# fig.show()
 # Linear Regression
# Preprocessing

# Splitting the data into training and test sets
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
print(len(train_set))
print(len(test_set))

wine = train_set.copy()
print(wine.head(100))
wine = train_set.drop("1", axis=1)
wine_labels = train_set["1"].copy()
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
wine_test = test_set.drop("1", axis=1)
wine_test_labels = test_set["1"].copy()
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
wine_test = test_set.drop("1", axis=1)
wine_test_labels = test_set["1"].copy()
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
wine_test = test_set.drop("1", axis=1)
wine_test_labels = test_set["1"].copy()
wine_test_prepared = full_pipeline.transform(wine_test)
tree_pred = tree_reg.predict(wine_test_prepared)
tree_mse = mean_squared_error(wine_test_labels, tree_pred)
tree_rmse = np.sqrt(tree_mse)
print(f"Decision Tree Regression RMSE for Test Set: {tree_rmse}")









# For this exercise, I watched the video lectures that you provided in the WyoCourse once again to learn more about the concepts of regression and classification
# I also read some of the materials on the web about how to code nicely in machine learning and python
# But mostly I read two chapters of the book named "Hands-On Machine Learning with scikit-learn and tensorflow"
# I reallu learned a lot by reading this book, especially the concepts in preprocessing section. That was so practical
# As the results show, for Linear Regression the error is 0.2 which is high, partly because I chose a model (LR) which is not best fit for this problem
# becuase label valuse are not continuous - The test error for linear regression is 0.3
# But in logistic regression, which is good for classification, the test error is 0.3
# For Decision Tree Regression, the test error is 0.4
# We cannot expect to have good results because I think we have short in the number of data!
