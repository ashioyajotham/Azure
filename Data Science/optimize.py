import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/daily-bike-share.csv
bike_data = pd.read_csv("daily-bike-share.csv")
bike_data["day"] = pd.DateTimeIndex(bike_data["dteday"]).day
numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
categorical_features = ['season','mnth','holiday','weekday','workingday','weathersit', 'day']
bike_data[numeric_features + ['rentals']].describe()
print(bike_data.head())

"""
Separate features and labels
After separating the dataset, we now have numpy arrays named **X** containing the features, and **y** containing the labels.
"rentals" is our target_column
"""
X,y = bike_data[["season","month","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed"]].values, bike_data.rentals.values

# Split data 70%-30% into training set and test set:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
print("Training set: {}\nTest set: {}").format(X_train.shape[0], X_test.shape[0]) #Str.format()


"""
Now we have the following four datasets:

* X_train: The feature values we'll use to train the model
* y_train: The corresponding labels we'll use to train the model
* X_test: The feature values we'll use to validate the model
* y_test: The corresponding labels we'll use to validate the model
"""


"""
Now we're ready to train a model by fitting a boosting ensemble algorithm 
recall that a Gradient Boosting estimator, is like a Random Forest algorithm, but instead of building them all trees independently and taking the average result, 
each tree is built on the outputs of the previous one in an attempt to incrementally reduce the loss (error) in the model.
"""

#Train the model
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

#Fit a Lasso Model on the training set
model = GradientBoostingRegressor().fit(X_test, y_test)
print("model",\n)

#Evaluate the model using test data
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:",rmse)
r2 = r2_score(y_test, predictions)
print("R2:",r2)

# Plot predicted vs actual
plt.scatter(y_test, predictions)
plt.xlabel("Actual Labels")
plt.ylabel("Predicted Labels")
plt.title("Daily Bike Share Predictions")

#Overlay regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color="magenta")
plt.show()


## Optimize Hyperparameters
"""
In machine learning, the term parameters refers to values that can be determined from data; 
values that you specify to affect the behavior of a training algorithm are more correctly referred to as hyperparameters.

The specific hyperparameters for an estimator vary based on the algorithm that the estimator encapsulates. 
In the case of the GradientBoostingRegressor estimator, the algorithm is an ensemble that combines multiple decision trees to create an overall predictive model

Let us try the GridSearch
"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score

# Use a Gradient Boosting algorithm
alg = GradientBoostingRegressor()

# Try these hyperparameter values
params = {
 'learning_rate': [0.1, 0.5, 1.0],
 'n_estimators' : [50, 100, 150]
 }

# Find the best hyperparameter combination to optimize the R2 metric
score = make_scorer(r2_score)
gridsearch = GridSearchCV(alg, params, scoring=score, cv=3, return_train_score=True)
gridsearch.fit(X_train, y_train)
print("Best parameter combination:", gridsearch.best_params_, "\n")

# Get the best model
model=gridsearch.best_estimator_
print(model, "\n")

# Evaluate the model using the test data
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, predictions)
print("R2:", r2)

# Plot predicted vs actual
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')
# overlay the regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()

"""
Machine learning models work best with numeric features rather than text values, so you generally need to convert categorical features into numeric representations. 
For example, suppose your data includes the following categorical feature. You can apply ordinal encoding to substitute a unique integer value for each category.
Another common technique is to use one hot encoding to create individual binary (0 or 1) features for each possible category value. 
For example, you could use one-hot encoding to translate the possible categories into binary columns

To apply these preprocessing transformations to the bike rental, we'll make use of a Scikit-Learn feature named pipelines. 
These enable us to define a set of preprocessing steps that end with an algorithm. You can then fit the entire pipeline to the data, so that the model encapsulates all of the preprocessing steps as well as the regression algorithm. 
This is useful, because when we want to use the model to predict values from new data, we need to apply the same transformations (based on the same statistical distributions and category encodings used with the training data).
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklean.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
import numpy as np

#Define prepocessing for numeric columns(scale them)
numeric_features = [6,7,8,9]
numeric_transformer = Pipeline(steps=[
  ('scaler', StandardScaler())])

#Define preprocessing for categorical features
categorical_features = [0,1,2,3,4,5]
categorical_transformer = Pipeline(steps=[
  ("onehot", OneHotEncoder(handle_unknown="ignore"))])

#Combine preprocessing types
preprocessor = ColumnTransformer(
  transformers = [
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)])

#Create preprocessing and training pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("regressor", GradientBoostingRegressor())])

# fit the pipeline to train a linear regression model on the training set
model = pipeline.fit(X_train, (y_train))
print (model)


## Validation
### Get Predictions
predictions = model.predict(X_test)

###Display Metrics
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, predictions)
print("R2:", r2)

# Plot predicted vs actual
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()

"""
The pipeline is composed of the transformations and the algorithm used to train the model. 
To try an alternative algorithm you can just change that step to a different kind of estimator.
"""
# Use a different estimator in the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor())])


# fit the pipeline to train a linear regression model on the training set
model = pipeline.fit(X_train, (y_train))
print (model, "\n")

# Get predictions
predictions = model.predict(X_test)

# Display metrics
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, predictions)
print("R2:", r2)

# Plot predicted vs actual
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions - Preprocessed')
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()

"""
Use the Trained Model
First, let's save the model.
"""
import joblib

# Save the model as a pickle file
filename = './bike-share.pkl'
joblib.dump(model, filename)

# Now, we can load it whenever we need it, and use it to predict labels for new data. This is often called scoring or inferencing.

# Load the model from the file
loaded_model = joblib.load(filename)

# Create a numpy array containing a new observation (for example tomorrow's seasonal and weather forecast information)
X_new = np.array([[1,1,0,3,1,1,0.226957,0.22927,0.436957,0.1869]]).astype('float64')
print ('New sample: {}'.format(list(X_new[0])))

# Use the model to predict tomorrow's rentals
result = loaded_model.predict(X_new)
print('Prediction: {:.0f} rentals'.format(np.round(result[0])))



# An array of features based on five-day weather forecast
X_new = np.array([[0,1,1,0,0,1,0.344167,0.363625,0.805833,0.160446],
                  [0,1,0,1,0,1,0.363478,0.353739,0.696087,0.248539],
                  [0,1,0,2,0,1,0.196364,0.189405,0.437273,0.248309],
                  [0,1,0,3,0,1,0.2,0.212122,0.590435,0.160296],
                  [0,1,0,4,0,1,0.226957,0.22927,0.436957,0.1869]])
result = loaded_model(X_new)
print("5-day rental predictions")
for prediction in results:
  print(np.round(prediction))

