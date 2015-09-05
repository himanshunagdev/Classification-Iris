# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 21:00:29 2015

@author: NAGDEV
"""
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# load the iris dataset
irisDataset= datasets.load_iris()
#apply Logistic regression
classDataset_LR = LogisticRegression()

#test and train
classDataset_LR.fit(irisDataset.data, irisDataset.target)

tgtExpected = irisDataset.target
pred = classDataset_LR.predict(irisDataset.data)

#check fit of the model
print(metrics.classification_report(tgtExpected, pred))
print(metrics.confusion_matrix(tgtExpected, pred))

###############################################################################################

# Support Vector Machine classification
#apply Support Vector Machine classification
classData_SVC = SVC()
#test and train
classData_SVC.fit(irisDataset.data, irisDataset.target)

tgtExpected_SVC = irisDataset.target
pred_SVC = classData_SVC.predict(irisDataset.data)
# check the fit of the model
print(metrics.classification_report(tgtExpected_SVC, pred_SVC))
print(metrics.confusion_matrix(tgtExpected_SVC, pred_SVC))


###############################################################################################

#Desicion TREE
# fiting a CART( Classification and Regression Tree ) to the data
classData_DT = DecisionTreeClassifier()
#test and train
classData_DT.fit(irisDataset.data, irisDataset.target)

# predictions
tgtExpected_DT = irisDataset.target
pred_DT = classData_DT.predict(irisDataset.data)

# summarize the fit of the model
print(metrics.classification_report(tgtExpected_DT, pred_DT))
print(metrics.confusion_matrix(tgtExpected_DT, pred_DT))