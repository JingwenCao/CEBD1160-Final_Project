# 1. Import all libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as ppt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# 2. Load data
breastcancer = load_breast_cancer()
X, headers, y = breastcancer.data, breastcancer.feature_names, breastcancer.target

# 3. Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Split data into training data and test data
X_train, X_test, Y_train, Y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)

# 3a. Make data two dimensional
pca_model = PCA(n_components=2)
pca_model.fit(X_train)
X_train = pca_model.transform(X_train)
X_test = pca_model.transform(X_test)
X_train[:5]


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
ppt.savefig ("Classifiers_Performance.png", format="PNG")

# 6. Plot all classification plots
fig = ppt.figure(figsize=(27,27))

# 6a. Plot three attributes that were my GOAT
matplotlib.style.use('ggplot')
h = 0.02 

for i in [1,2,3,4,5,6,7,8,9]:
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                     np.arange(y_min, y_max, h))
	cm = ppt.cm.RdBu
	cm_bright = ListedColormap(['#FF0000', '#0000FF'])
	ax = ppt.subplot(3,3,i)
	ax.set_title("Input Data")
	plot1=ax.scatter(X_train[:,0], X_train[:,1], c=Y_train, cmap=cm_bright, edgecolors='k')
	plot2=ax.scatter(X_test[:,0], X_test[:,1], c=Y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
	ax.legend((plot1, plot2), ('Training Data', 'Test Data'))
	ax.set_xlim(xx.min(), xx.max())
	ax.set_ylim(yy.min(), yy.max())
	ax.set_xticks(())
	ax.set_yticks(())

# 6b. Plot decision boundary function
def plot_decision_boundary(model,data,position):
    ax = ppt.subplot(3,3,position)
    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)

    if hasattr(model, "decision_function"):
    	Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8, norm=None)

    ax.scatter(X_train[:,0], X_train[:,1], c=Y_train, cmap=cm_bright, edgecolors='k')
    ax.scatter(X_test[:,0], X_test[:,1], c=Y_test, cmap=cm_bright, alpha=0.3, edgecolors='k')

    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(cla_names[i])

for i in (range(len(classifiers))):
	plot_decision_boundary(classifiers[i], X, i+2)
ppt.tight_layout()
ppt.savefig ("Classifiers_Plots.png", format="PNG")