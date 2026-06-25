#Data augmentation: for each image in the training set, add 4 shifted copies of it

from scipy.ndimage import shift
import matplotlib.pyplot as plt
import numpy as np

## mnist is a database of digits
from sklearn.datasets import fetch_openml
mnist=fetch_openml("mnist_784", as_frame=False) #False to get the dataset as NumPy arrays
X,y=mnist.data, mnist.target

#create a train set and a test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#an image is e.g. X_train[0]
def shift_up(image_data):
    image_2d = image_data.reshape(28,28)
    shifted_image_2d=shift(image_2d, [-1, 0], cval=0, order=0)
    #return the "flattened" image
    return shifted_image_2d.reshape(-1)

def shift_down(image_data):
    image_2d = image_data.reshape(28,28)
    shifted_image_2d=shift(image_2d, [1, 0], cval=0, order = 0)
    #return the "flattened" image
    return shifted_image_2d.reshape(-1)

def shift_right(image_data):
    image_2d = image_data.reshape(28,28)
    shifted_image_2d=shift(image_2d, [0, 1], cval=0, order = 0)
    #return the "flattened" image
    return shifted_image_2d.reshape(-1)

def shift_left(image_data):
    image_2d = image_data.reshape(28,28)
    shifted_image_2d=shift(image_2d, [0, -1], cval=0, order =0)
    #return the "flattened" image
    return shifted_image_2d.reshape(-1)

def shift_copies(image_data):
    return shift_up(image_data), shift_down(image_data), shift_left(image_data), shift_right(image_data)


#For each image_data in X_train, we create 4 copies and append them to X_train
#They all belong to the same class, thus we append four times the same element to y_train
# Create lists to hold the augmented data
X_train_augmented = []
y_train_augmented = []

# Loop through every image and label in the original training set
for image, label in zip(X_train, y_train):
    # Add the original image and its label
    X_train_augmented.append(image)
    y_train_augmented.append(label)
    
    # Generate the 4 shifted copies
    up, down, left, right = shift_copies(image)
    
    # Use .extend() to add the 4 individual arrays into the main list
    X_train_augmented.extend([up, down, left, right])
    
    # 4. Add the same label 4 more times to match the 4 shifts
    y_train_augmented.extend([label] * 4)

# Convert the final lists back into fast NumPy arrays for scikit-learn
X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)


from sklearn.neighbors import KNeighborsClassifier

#### ----- The model
#from ex1 we know the optimal parameters
# Added n_jobs=-1 so your computer processes the calculations in parallel
best_knn_model = KNeighborsClassifier(n_neighbors=4, weights="distance", n_jobs=-1)
#train the model on the augmented training set
best_knn_model.fit(X_train_augmented, y_train_augmented)

### -------  Error analysis on the test set
from sklearn.metrics import accuracy_score

# Generate predictions on the augmented training set
y_test_pred = best_knn_model.predict(X_test)

# Compute and print the final accuracy score
print(f"Accuracy On The Test Set: {accuracy_score(y_test, y_test_pred):.2%}")
#-----> 97.63%







