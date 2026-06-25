#the goal of the exercise is to build a classifier for the mnist dataset that achieves over 97% accuracy
#Hint: KNeighborsClassifier

import matplotlib.pyplot as plt

## mnist is a database of digits
from sklearn.datasets import fetch_openml
mnist=fetch_openml("mnist_784", as_frame=False) #False to get the dataset as NumPy arrays
X,y=mnist.data, mnist.target


#create a train set and a test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier()
#from below we now know that:
best_knn_model=KNeighborsClassifier(n_neighbors=4, weights="distance")

# Train the model
best_knn_model.fit(X_train, y_train)

### -------  Error analysis (on the original NOT augmented training set)
from sklearn.metrics import accuracy_score

# Generate predictions on the augmented training set
y_test_pred = best_knn_model.predict(X_test)
print(f"Accuracy On The Test Set: {accuracy_score(y_test, y_test_pred):.2%}")
#----> answer 97.14%


### -------  APPROACH 1  measure accuracy using cross validation
"""from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X_train, y_train, cv=3, scoring="accuracy")

#Print the individual fold scores
print("Scores per fold:", scores)

#Print the average score across all folds
print("Mean accuracy:", scores.mean())"""

"""### -------  Error analysis
from sklearn.metrics import ConfusionMatrixDisplay

#first make predictions
from sklearn.model_selection import cross_val_predict
y_train_pred=cross_val_predict(knn, X_train, y_train, cv=3)
# normalize the confusion matrix 
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, normalize="true", values_format=".0%")
plt.show()"""

"""### ------ APPROACH 2: grid search (warning, we found optimal parameters so no need to run the grid search again)
from sklearn.model_selection import GridSearchCV

# Define the grid of parameters to test
param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]

grid_search = GridSearchCV(knn, param_grid, cv=3, scoring="accuracy")

# Run the search on the training data
grid_search.fit(X_train, y_train)

# See the best hyperparameter combination found
print("Best parameters:", grid_search.best_params_)
####  use this model!
#Best parameters: {'n_neighbors': 4, 'weights': 'distance'}
#best_knn_model=KNeighborsClassifier(n_neighbors=4, weights="distance")

# See the highest accuracy score (should be >97%)
print("Best cross-validation score:", grid_search.best_score_)
#Best cross-validation score: 0.9703500000000002

# Extract the ready-to-use model with the absolute best parameters (use best_knn_model from above, no need for grid search)
best_knn_model = grid_search.best_estimator_

### -------  Error analysis
from sklearn.metrics import ConfusionMatrixDisplay

#first make predictions (we already have a mode, no need to re-train it another time: no need for cross_val_predict)

y_train_pred=best_knn_model.predict(X_train)
# normalize the confusion matrix 
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, normalize="true", values_format=".0%")
plt.show()"""







