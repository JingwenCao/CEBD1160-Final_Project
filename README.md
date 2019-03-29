# CEBD1160 - Wisconsin Breast Cancer

| Name | Date |
|:-------|:---------------|
|Jingwen Cao | March 23, 2019|

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

4 sentence longer explanation about your research question. Include:

- opportunity (what data do we have)
- challenge (what is the "problem" we could solve with this dataset)
- action (how will we try to solve this problem/answer this question)
- resolution (what did we end up producing)

Greg Sample:

Derived from Diffusion-Weighted Magnetic Resonance Imaging (DWI, d-MRI), we have derived "maps" of structural connectivity between brain regions. Using these data, we may be able to understand relationships between brain regions and their relative connectivity, which can then be used for targetted interventions in neurodegenerative diseases. Here, we tried to predict the connectivity between two unique brain regions based on all other known brain connectivity maps. Based on the preliminary performance of this regressor, we found that the current model didn't provide consistent performance, but shows promise for success with more sophisticated methods.

### Introduction

Whilst there are several ways to determine whether breast cancer cells are cancerous or not, one of the less invasive ways is via FNA (fine needle aspiration), wherein a thin, hollow needle is used to withdraw tissue or fluid from a suspicious area [1]. The Wisconsin Breast Cancer Data Set consists of 569 instances of digitized images of breast masses collected by FNA, and is relatively clean and noise-free. Each image is tagged with an ID number, as well as its diagnosis (benign or malignant), and the means, standard errors, and “worst” or largest values of ten features are described for each image [2]. Thus, this dataset is well suited to try to assess which machine-learning model best predicts whether breast mass cells are benign or malignant. The analysis will be conducted by Jingwen Cao, using the following libraries: Sklearn, Matplotlib, ____________, ___________. The graphs represent _____________.

#### Sources:
- *[1] FNA (https://www.cancer.org/cancer/breast-cancer/screening-tests-and-early-detection/breast-biopsy/fine-needle-aspiration-biopsy-of-the-breast.html)*
- *[2] Wisconsin Breast Cancer Data Set (https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names)*

### Methods

All supervised classification training models from the Scikit Learn resource for generalized linear models were selected to be assessed to determine the model that most accurately predicted whether a sample was malignant or benign [4]. These include: Logistic Regression, Support Vector Machines, Stochastic Gradient Descent, Nearest Neighbor, Naïve Bayes, Decision Trees, and Ensemble Methods. All data was split into two groups: Training data and test data, and subsequently standardized, as they varied drastically in magnitude [3]. All pseudocode can be found here (hyperlink). Accuracy scores for all models were plotted in a bar graph, as shown in Figure 1 below.

#### Sources:
- *[3] Normalizing data (https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029)*
- *[4] Supervised Training models (https://scikit-learn.org/stable/supervised_learning.html)*
- *[5] https://markd87.github.io/articles/ml.html
- *[6] https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
- *[7] http://benalexkeen.com/k-nearest-neighbours-classification-in-python/

##### Figure 1
![Figure 1](https://raw.githubusercontent.com/JingwenCao/CEBD1160-Final_Project/master/Classifiers_Performance.png)


### Results

The supervised classification model that most accurately predicted whether the test samples were benign were malignant was ____________. Figure 2 below shows the performance on the test set.
Figure 2

[Synopsis on the diagram and performance of the model]

Brief (2 paragraph) description about your results. Include:

- At least 1 figure
- At least 1 "value" that summarizes either your data or the "performance" of your method
- A short explanation of both of the above

Greg example:
We can see that in general, our regressor seems to underestimate our edgeweights. In cases where the connections are small, the regressor performs quite well, though in cases where the strength is higher we notice that the performance tends to degrade.

### Discussion
Brief (no more than 1-2 paragraph) description about what you did. Include:

- interpretation of whether your method "solved" the problem
- suggested next step that could make it better.

Greg example:
The method used here does not solve the problem of identifying the strength of connection between two brain regions from looking at the surrounding regions. This method shows that a relationship may be learnable between these features, but performance suffers when the connection strength is towards the extreme range of observed values. To improve this, I would potentially perform dimensionality reduction, such as PCA, to try and compress the data into a more easily learnable range.

### References
All of the links

-------
