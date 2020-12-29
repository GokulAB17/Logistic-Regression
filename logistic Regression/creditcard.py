#Classify whether application accepted or not using Logistic regression
#card - Factor. Was the application for a credit card accepted?
#reports - Number of major derogatory reports.
#age - Age in years plus twelfths of a year.
#income -Yearly income (in USD 10,000).
#share - Ratio of monthly credit card expenditure to yearly income.
#expenditure - Average monthly credit card expenditure.
#owner - Factor. Does the individual own their home?
#selfemp -Factor. Is the individual self-employed?
#dependents - Number of dependents.
#months - Months living at current address.
#majorcards - Number of major credit cards held.
#active - Number of active credit accounts.

import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt

#Loading Data set creditcard.csv in python environment
cc= pd.read_csv(r"file-path\creditcard.csv",index_col=0)

#to view features of data set
cc.columns

#Data Preprocessing for model building
#creating dummy variables for all categorical features
cc.card=cc.card.map({"yes":1,"no":0})
cc.owner=cc.owner.map({"yes":1,"no":0})
cc.selfemp=cc.selfemp.map({"yes":1,"no":0})

#checking null values or na values
cc.isnull().sum()

#Model Building
import statsmodels.formula.api as sm
#Singular matrix error while building model by taking all features
logit_model = sm.logit('card~share+expenditure+reports+age+income+owner+selfemp+dependents+months+majorcards+active',data = cc).fit()

#finding the features with high correlations in between
#share and expenditure have high correlation
from scipy.stats import pearsonr
s=cc["share"]
e=cc["expenditure"]
corr,_ = pearsonr(s, e) # 0.83877 high correlation 

#Removing expenditure feature from model building to avoid error
logit_model = sm.logit('card~share+reports+age+income+owner+selfemp+dependents+months+majorcards+active',data = cc).fit()

#summary and predictions
logit_model.summary()
card_pred = logit_model.predict(cc) 


cc["pred_prob"] = card_pred
# Creating new column for storing predicted class of card

# filling all the cells with zeroes
cc["card_val"] = np.zeros(1319)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
cc.loc[card_pred>=0.5,"card_val"] = 1
cc.card_val

from sklearn.metrics import classification_report
classification_report(cc.card_val,cc.card)

# confusion matrix 
confusion_matrix = pd.crosstab(cc['card'],cc.card_val)

confusion_matrix
accuracy = (294+1000)/(1319) # 89.57
accuracy

# ROC curve 
from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(cc.card, card_pred)


# the above function is applicable for binary classification class 

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
 
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 
#roc_auc=0.9966

### Dividing data into train and test data sets
cc.drop("card_val",axis=1,inplace=True)
from sklearn.model_selection import train_test_split

train,test = train_test_split(cc,test_size=0.3)

# checking na values 
train.isnull().sum();test.isnull().sum()

# Building a model on train data set 

train_model = sm.logit('card~share+reports+age+income+owner+selfemp+dependents+months+majorcards+active',data = train).fit()

#summary
train_model.summary()
train_pred = train_model.predict(train)

# Creating new column for storing predicted class of y

# filling all the cells with zeroes
train["train_pred"] = np.zeros(923)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
train.loc[train_pred>0.5,"train_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(train['card'],train.train_pred)

confusion_matrix
accuracy_train = (201+699)/(923) # 97.51
accuracy_train

# Prediction on Test data set

test_pred = train_model.predict(test)

# Creating new column for storing predicted class of y

# filling all the cells with zeroes
test["test_pred"] = np.zeros(396)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
test.loc[test_pred>0.5,"test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test['card'],test.test_pred)

confusion_matrix
accuracy_test = (88+302)/(396) # 98.48
accuracy_test
