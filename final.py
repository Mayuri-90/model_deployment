## You have to build a solution that should able to predict the premium of the personal
# for health insurance
# Import the libraries
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline

# Linear regression import
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Import the Evaluation metrics
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import pickle

##Loading and preprocessing of data¶
data = pd.read_csv('insurance.csv')
data.head()
# Number of rows and column
data.shape
# Getting some information about the dataset
data.info()
#Checking for missing values
data.isnull().sum()

# Data Analysis
# distribution of age value
#Gender countplot
print(data['sex'].value_counts())
plt.figure(figsize=(6,6))
sns.countplot(data['sex'])
plt.title('Sex Distribution')
plt.show()

# Children countplot
print(data['children'].value_counts())
plt.figure(figsize=(6,6))
sns.countplot(data['children'], color = 'b')
plt.title('children Distribution')
plt.show()

#smoker column
print(data.smoker.value_counts())

plt.figure(figsize=(6,6))
sns.countplot(data.smoker)
plt.title('Smoker')
plt.show()

#region column
print(data.region.value_counts())
sns.countplot(data.region,color = 'lightgreen')
plt.show()

sns.boxplot(data.expenses)

#split the data
x_train = data.iloc[:,:6] # Independent Variables

y_train = data.iloc[:,6]   # 6 is the index of "expeses" in the training data Set , setting it as the label column

# Scaling Data
# Creating new variables for numerical attributes/columns
numeric_features = ['age', 'bmi', 'children']
# Making pipeline for scaling down numerical features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# Creating new variables for categorical attributes/columns
categorical_features = ['sex', 'smoker','region']
# MAking pipeling for Encoding categorical features
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Creating new variable for these numerical & categorical features pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x_train,y_train, test_size=0.2, random_state = 0)

#Linear Regression

linear_regressor =  Pipeline(steps = [('preprocessor',preprocessor),('linear_regressor', LinearRegression())])
linear_regressor.fit(X_train,Y_train)
#Predict the Model
y_pred = linear_regressor.predict(X_test)

#Evaluation Metrics
linear_regressor_mse = mean_squared_error(y_pred,Y_test)
linear_regressor_rmse = np.sqrt(linear_regressor_mse)
linear_regressor_r2_score =r2_score(y_pred,Y_test)

print('linear_regressor_MSE: ',linear_regressor_mse)
print('linear_regressor_RMSE: ',linear_regressor_rmse)
print('linear_regressor_score:',linear_regressor_r2_score )

#Decision Tree
decisio_tree_regressor = Pipeline(steps=[('preprocessor', preprocessor),
                                         (
                                         'DecisionTreeRegressor', DecisionTreeRegressor(max_depth=5, random_state=13))])

# fit the regressor with X and Y data
decisio_tree_regressor.fit(X_train, Y_train)

# y_pred_dt = regressor1.predict(X_test)

y_pred1 = decisio_tree_regressor.predict(X_test)
decisio_tree_regressor_MSE = mean_squared_error(y_pred1, Y_test)
decisio_tree_regressor_RMSE = np.sqrt(decisio_tree_regressor_MSE)
decisio_tree_regressor_r2_score = r2_score(Y_test, y_pred1)

print("decisio_tree_regressor_MSE : ", decisio_tree_regressor_MSE)
print('decisio_tree_regressor_RMSE: ', decisio_tree_regressor_RMSE)
print('decisio_tree_regressor_r2_score ', decisio_tree_regressor_r2_score)

#Random Forest
random_forest_regressor =Pipeline(steps = [('preprocessor',preprocessor),
                                     ('RandomForestRegressor',
                                      RandomForestRegressor(n_estimators=400, max_depth=5, random_state=13))])
random_forest_regressor.fit(X_train, Y_train)

y_pred2 = random_forest_regressor.predict(X_test)

random_forest_regressor_MSE = mean_squared_error(y_pred2,Y_test)
random_forest_regressor_RMSE = np.sqrt(random_forest_regressor_MSE)
random_forest_regressor_MAE = mean_absolute_error(y_pred2,Y_test)
random_forest_regressor_r2_score = r2_score(Y_test,y_pred2)

print("random_forest_regressor_MSE : ",random_forest_regressor_MSE)
print("random_forest_regressor_RMSE :", random_forest_regressor_RMSE)
print("random_forest_regressor_MAE : ",random_forest_regressor_MAE )
print("random_forest_regressor_r2_score :",random_forest_regressor_r2_score )


#Gradient_Boosting_Regressor
# Testing with GradientBoostingRegressor
Gradient_Boosting_Regressor= Pipeline(steps = [('preprocessor',preprocessor),
                        ('GradientBoostingRegressor',GradientBoostingRegressor(max_depth=2, n_estimators=100, learning_rate =.2))])
Gradient_Boosting_Regressor.fit(X_train,Y_train)

#Predict the model
y_pred3 = Gradient_Boosting_Regressor.predict(X_test)

#Evaluation Metrics
Gradient_Boosting_Regressor_MSE = mean_squared_error(y_pred3,Y_test)
Gradient_Boosting_Regressor_RMSE = np.sqrt(Gradient_Boosting_Regressor_MSE)
Gradient_Boosting_Regressor_r2_score = r2_score(Y_test,y_pred3)


print("Gradient_Boosting_Regressor_MSE : ",Gradient_Boosting_Regressor_MSE)
print('Gradient_Boosting_Regressor_RMSE: ',Gradient_Boosting_Regressor_RMSE)
print('Gradient_Boosting_Regressor_r2_score ',Gradient_Boosting_Regressor_r2_score )

models = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
    'RMSE': [linear_regressor_rmse, decisio_tree_regressor_RMSE, random_forest_regressor_RMSE,
             Gradient_Boosting_Regressor_RMSE],

    'r2_score': [linear_regressor_r2_score, decisio_tree_regressor_r2_score, random_forest_regressor_r2_score,
                 Gradient_Boosting_Regressor_r2_score]})

models.sort_values(by='r2_score', ascending=False)

pickle.dump(Gradient_Boosting_Regressor,open('insurance_premium_prediction_model.pkl','wb'))