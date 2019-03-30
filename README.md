# CEBD1160 - Wisconsin Breast Cancer

| Name | Date |
|:-------|:---------------|
|Jingwen Cao | March 29, 2019|

-----

### Resources
Your repository should include the following:

- Python script for your analysis: breastcancer_analysis.py
- Results figure/saved file: figures/
- Dockerfile for your experiment: Dockerfile
- runtime-instructions in a file named RUNME.md

-----

## Research Question

Which machine-learning model best predicts whether a tumor is benign or malignant?

### Abstract

Using Fine Needle Aspiration, 30 samples of breast cancer cells were collected by Dr. William H. Wolberg, from the University of Wisconsin. Using this data, we tried to determine the machine learning algorithm that would best predict whether a sample was benign or malignant, based on the features provided. This was done by selecting 8 classification models, and plotting their respective contours and scores over the training and testing data. From the preliminary analysis, we see that the best algorithm was the Logistic Regression model, though further analysis should be done to manipulate the parameters of each model.

### Introduction

Whilst there are several ways to determine whether breast cancer cells are cancerous or not, one of the less invasive ways is via FNA (fine needle aspiration), wherein a thin, hollow needle is used to withdraw tissue or fluid from a suspicious area [1]. The Wisconsin Breast Cancer Data Set consists of 569 instances of digitized images of breast masses collected by FNA, and is relatively clean and noise-free. Each image is tagged with an ID number, as well as its diagnosis (benign or malignant), and the means, standard errors, and “worst” or largest values of ten features are described for each image [2]. Thus, this dataset is well suited to try to assess which machine-learning model best predicts whether breast mass cells are benign or malignant. The analysis will be conducted by Jingwen Cao, using the following libraries: Sklearn, Matplotlib, and Numpy. The graphs represent the performance and visualizations for the various models applied to the data.

#### Sources:
- *[1] FNA (https://www.cancer.org/cancer/breast-cancer/screening-tests-and-early-detection/breast-biopsy/fine-needle-aspiration-biopsy-of-the-breast.html)*
- *[2] Wisconsin Breast Cancer Data Set (https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names)*

### Methods

All supervised classification training models from the Scikit Learn resource for generalized linear models were selected to be assessed to determine the model that most accurately predicted whether a sample was malignant or benign [4]. These include: Logistic Regression, Support Vector Machines, Stochastic Gradient Descent, Nearest Neighbor, Naïve Bayes, Decision Trees, and Ensemble Methods. All data was split into two groups: Training data and test data, and subsequently standardized, as they varied drastically in magnitude [3]. All pseudocode can be found here (hyperlink). Accuracy scores for all models were plotted in a bar graph, as shown in Figure 1 below.

#### Sources:
- *[3] Normalizing data (https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029)*
- *[4] Supervised Training models (https://scikit-learn.org/stable/supervised_learning.html)*

##### Figure 1
![Figure 1](https://raw.githubusercontent.com/JingwenCao/CEBD1160-Final_Project/master/Classifiers_Performance.png)


### Results

The supervised classification model that most accurately predicted whether the test samples were benign were malignant was the Logistic Regression model. Figure 2 below shows the performance on the test set.

##### Figure 2
![Figure 2](https://github.com/JingwenCao/CEBD1160-Final_Project/blob/master/Classifiers_Plots.png)

The Logistic Regression model was the most accurate model (0.965) by a very small margin, over the Nearest Neighbours model (0.961). Overall, all regressors performed fairly similarly, with no significant differences between models. One model trending towards a significant difference was the Naive Bayes model as it seemed to struggle to handle the cases where the two samples overlapped.

### Discussion
All classification models were used with their default settings. While this gave us a preliminary idea of which algorithm performed well or not, tweaking the parameters could change their overall performance. None of the models breached a performance of 0.96, which means there is room for improvement. Thus, more in depth testing, with manipulation of all sensible parameters, needs to be done to determine the absolute best model. After all, when it comes to live diagnosis, a false negative, and even a false positive, can have very a significant and negative impact on the patients.

### References
Heavy inspiration drawn from:
- *[5] https://markd87.github.io/articles/ml.html*
- *[6] https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html*
- *[7] http://benalexkeen.com/k-nearest-neighbours-classification-in-python/*
- *[8] https://github.com/gkiar/cebd1160_project_template/tree/gkexample*

-------
