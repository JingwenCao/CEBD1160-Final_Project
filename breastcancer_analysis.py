# 1. Import all libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as ppt
from matplotlib.colors import ListedColormap
from argparse import ArgumentParser
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

# 2. Load data
breastcancer = load_breast_cancer()
data, headers, target = breastcancer.data, breastcancer.feature_names, breastcancer.target

# 3. Split data into training data and test data
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, target, test_size=0.25, random_state=0)

# 4. Standardize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Perform all classifications using various models
# 5a. Define all models
cla_names = ["Logistic Regression", "SVM", "Stochastic Gradient Descent", "Nearest Neighbors", "Naive Bayes", "Decision Trees", "Bagging Meta-Estimator", "Random Forest"]
classifiers = [
    LogisticRegression(random_state=0), 
    SVC(kernel='rbf', random_state=0), 
    SGDClassifier(loss="hinge", penalty='l2', max_iter=5), 
    KNeighborsClassifier(n_neighbors=7), 
    GaussianNB(), 
    DecisionTreeClassifier(random_state=0), 
    BaggingClassifier(random_state=0), 
    RandomForestClassifier(random_state=0)]

# 5b. Apply all models in for loop
name_list = []
score = []
for name, clf in zip(cla_names, classifiers):
    clf.fit(X_train, Y_train)
    name_list.append(name)
    score.append(clf.score(X_test, Y_test))
# 5c Plot performance
fig, ax = ppt.subplots(figsize=(20, 10))
ind = np.arange(8)
width = 0.9
ppt.bar(ind, score, width, 0)
ppt.xticks(ind, name_list)
ppt.title("Classifier Performance")
i=0
for i in range(len(score)):
    ppt.annotate(score[i], xy=(i-0.4,0.99))
    i = i+1
ppt.savefig ("ClassifiersPerformance.png", format="PNG")

# 6. Plot all classification plots
fig = ppt.figure(figsize=(27,27))

# 6a. Plot three attributes that were my GOAT
benign_rows = data[breastcancer.target == 0]
malignant_rows = data[breastcancer.target == 1]

matplotlib.style.use('ggplot')

ax = fig.add_subplot(3,3,1, projection='3d')
line1 = ax.scatter(benign_rows[:,1], benign_rows[:,3], benign_rows[:,7])
line2 = ax.scatter(malignant_rows[:,1], malignant_rows[:,3], malignant_rows[:,7])
ax.legend((line1, line2), ('Benign', 'Malignant'))
ax.set_xlabel('Mean Texture')
ax.set_ylabel('Mean Area')
ax.set_zlabel('Mean Concave Points')
ppt.savefig ("GOAT.png", format="PNG")

for i in [2,3,4,5,6,7,8,9]:
    x_min,x_max=data[:,1].min() - .5, data[:,1].max() + .5
    y_min,y_max=data[:,3].min() - .5, data[:,3].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
    cm = ppt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = ppt.subplot(3,3,i)
    ax.set_title(" Classification Graph")
    ax.scatter(X_train[:,1], X_train[:,3], c=Y_train, cmap=cm_bright, edgecolors='k')
    ax.scatter(X_test[:,1], X_test[:,3], c=Y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
    ax.set_xlim()
    ppt.savefig ("GOAT.png", format="PNG")
    i+=1

# 6b. Plot decision boundary function
def plot_decision_boundary(model,data,position):
    if hasattr(model, "decision_function"):
    	Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
    	Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]

    Z = Z.reshape(xx.shape)
    
    ax.contourf([xx, yy], Z, cmap=cm, alpha=.8)

    ax.scatter(X_train[:,1], X_train[:,3], c=Y_train, cmap=cm_bright, edgecolors='k')

    ax.scatter(X_test[:,1], X_test[:,3], c=Y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    ppt.tight_layout()
    ppt.savefig ("GOAT.png", format="PNG")

plot_decision_boundary(classifiers[3], X_train, 2)