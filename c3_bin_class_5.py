# Chapter 3 of Geron's book. Classification systems. Binary classification.

## mnist is a database of digits
from sklearn.datasets import fetch_openml
mnist=fetch_openml("mnist_784", as_frame=False) #False to get the dataset as NumPy arrays
X,y=mnist.data, mnist.target


### VISUALIZE ONE OF THE DIGITS
# each image, such as X[0], can be plotted as follows
# it is an array with 28*28 parameters, each being a number between 0 (white) and 255 (black)

'''import matplotlib.pyplot as plt
def plot_digit(image_data):
  image = image_data.reshape(28,28)
  plt.imshow(image, cmap="binary")
  plt.axis("off")

some_digit=X[0]
plot_digit(some_digit)
plt.show()'''

#create a train set and a test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#true for all 5 and false for all other digits
y_train_5 = (y_train == "5")
y_test_5 = (y_test == "5")  

### A binary classifier that classifies what is 5 and what is not 5
from sklearn.linear_model import SGDClassifier
import numpy as np


sgd_clf = SGDClassifier(random_state=42)
#we train the classifier on the whole training set and the new true/false labels
sgd_clf.fit(X_train, y_train_5)

#measure accuracy using cross validation
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
### Result: array([0.95035, 0.96035, 0.9604 ])
#This looks promising, however, compare to the following dummy classifier that simply
#classifies evry single image in the most frequent class, that is "non 5"
#so this classifier simply assigns to every image "non 5"
#since there are only about 10% of 5 in the training set, this dummy classifier is correct 90% of the time


from sklearn import dummy

from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")