# 1. Import all libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as ppt
from argparse import ArgumentParser
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

# 2. Load data
'breastcancer' = load breast cancer dataset from sklearn
'data' = data from breast cancer dataset
'headers' = feature names from breast cancer dataset
'target' = targets from breast cancer dataset

# 3. Standardize data
'scaler' = min max scaler function
'X' = fit to X data and transform it
'X' = transform X data

# 4. Split data into training data and test data
'X_train', 'X_test', 'Y_train', 'Y_test' = split data using the train_test_split function, with a test size 0.25 and random state 0

# 5. Perform all classifications using various models
# 5a. Define all models
'cla_names' = ["Logistic Regression", "SVM", "Stochastic Gradient Descent", "Nearest Neighbors", "Naive Bayes", "Decision Trees", "Bagging Meta-Estimator", "Random Forest"]
'classifiers' = [
	logistic regression (with random state 0)
	support vector classifier ()
	stochastic gradient descent classifier (with loss "hinge", pentalty "l2", and max_iter 5)
	neighbors classifier (with 7 neighbors)
	gaussian naive bayes classifier ()
	decision tree classififer (with random state 0)
	bagging classifier (with random state 0)
	random forest classifier (with random state 0)]

# 5b. Apply all models in for loop
for name, clf in zip(cla_names, classifiers):
	fit classifier with training data and targets
	'score' = classification score (test data, test targets)
# 5c Plot performance
	'fig', 'ax' = pyplot subplots ()
	pyplot bar ('classifiers', 'score')
	save fig ("ClassifiersPerofrmance.png", with PNG format)

# 6. Plot all classification plots
figure = plot figure (with figsize 9, 9)

# 6a. Plot three attributes that were my GOAT
'benign_rows' = data with target 0
'malignant_rows' = data with target 1

matplotlib use style "ggplot"

'fig' = pyplot figure ()
'ax' = 'fig' subplot (111, with 3d projections)
'line1' = scatterpot of texture, area, and concave points of 'benign_rows'
'line2' = scatterplot of texture, area, and concave points of 'malignant_rows'
set legend to 'line1', 'line2', with names 'Benign' and 'Malignant'
set x label to 'Texture'
set y label to 'Area'
set z label to 'Concave Points'

# 6b. Plot decision boundary function
define function 'plot_decision_boundary' with parameters classifier,X,y:
    padding=0.15
    res=0.01
    find x min and x max
    find y min and y max
    x_range=x_max-x_min
    y_range=y_max-y_min
    xx,yy= numpy meshgrid(np.arange(x_min,x_max,res),np.arange(y_min,y_max,res))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])   
    Z = Z.reshape(xx.shape)
    plot figure with figsize=(8,6)
    'cs' = plot contour(xx, yy, Z, cmap=plt.cm.Spectral)
    plot scatter(X[:,0], X[:,1], s=35, c=y, cmap=plt.cm.Spectral)

# 6c. Plot decision boundaries for all classifiers
for model in classifiers:
	plot_decision_boundary(model, x_train, y_train)