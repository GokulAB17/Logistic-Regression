#Attribute information For bank datasetInput variables:
# bank client data:
#1 - age (numeric)
#2 - job : type of job (categorical: "admin.","unknown","unemployed",
#"management","housemaid","entrepreneur","student",
#"blue-collar","self-employed","retired","technician","services") 
#3 - marital : marital status (categorical: "married","divorced","single";
# note: "divorced" means divorced or widowed)
#4 - education (categorical: "unknown","secondary","primary","tertiary")
#5 - default: has credit in default? (binary: "yes","no")
#6 - balance: average yearly balance, in euros (numeric) 
#7 - housing: has housing loan? (binary: "yes","no")
#8 - loan: has personal loan? (binary: "yes","no")
# related with the last contact of the current campaign:
#9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
#10 - day: last contact day of the month (numeric)
#11 - month: last contact month of year (categorical: "jan", "feb", "mar",
# ..., "nov", "dec")
#12 - duration: last contact duration, in seconds (numeric)
# other attributes:
#13 - campaign: number of contacts performed during this campaign and 
#for this client (numeric, includes last contact)
#14 - pdays: number of days that passed by after the client was last 
#contacted from a previous campaign (numeric, -1 means client was not previously contacted)
#15 - previous: number of contacts performed before this campaign and for this client (numeric)
#16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
#Output variable (desired target):
#17 - y - has the client subscribed a term deposit? (binary: "yes","no")
#8. Missing Attribute Values: None
#Output variable -> y
#y -> Whether the client has subscribed a term deposit or not 
#Binomial ("yes" or "no")

import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt

#loading bank-full.csv dataset in python environment
bankdata= pd.read_csv(r"file-path/bank-full.csv",sep=";")

#to view all features in the data set 
bankdata.columns

#Data Preprocessing for model building
#creating dummy variables for all categorical features manually not by labelencoder package
bankdata.job=bankdata.job.map({"admin.":1,"unknown":2,"unemployed":3,"management":4,"housemaid":5,"entrepreneur":6,"student":7,
"blue-collar":8,"self-employed":9,"retired":10,"technician":11,"services":12})
bankdata.marital=bankdata.marital.map({"married":1,"divorced":2,"single":3})
bankdata.education=bankdata.education.map({"unknown":1,"secondary":2,"primary":3,"tertiary":4})
bankdata.default=bankdata.default.map({"yes":1,"no":0})
bankdata.housing=bankdata.housing.map({"yes":1,"no":0})
bankdata.loan=bankdata.loan.map({"yes":1,"no":0})
bankdata.contact=bankdata.contact.map({ "unknown":1,"telephone":2,"cellular":3})
bankdata.month.unique()
bankdata.month=bankdata.month.map({"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                                   "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12})
bankdata.poutcome=bankdata.poutcome.map({"unknown":1,"other":2,"failure":3,"success":4})
bankdata.y=bankdata.y.map({"yes":1,"no":0})

#EDA
bankdata.describe()

#Checking no. of record entries
bankdata.shape

#checking null or na values in dataset
bankdata.isna().sum() # no na values

#Model building with all features
import statsmodels.formula.api as sm
logit_model = sm.logit('y~age+job+marital+education+default+balance+housing+loan+contact+month+duration+campaign+previous+poutcome',data = bankdata).fit()

#summary
logit_model.summary()

#predicting values
y_pred = logit_model.predict(bankdata)

bankdata["pred_prob"] = y_pred
# Creating new column for storing predicted class of y

# filling all the cells with zeroes
bankdata["y_val"] = np.zeros(45211)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
bankdata.loc[y_pred>=0.5,"y_val"] = 1
bankdata.y_val

from sklearn.metrics import classification_report
report=classification_report(bankdata.y_val,bankdata.y)

# confusion matrix 
confusion_matrix = pd.crosstab(bankdata['y'],bankdata.y_val)

confusion_matrix
accuracy = (39098+1399)/(45211) # 89.57
accuracy

# ROC curve 
from sklearn import metrics
help("metrics")
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(bankdata.y, y_pred)
#accuracy=0.8957

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
 
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 
#auc=0.878

### Dividing data into train and test data sets
bankdata.drop("y_val",axis=1,inplace=True)
from sklearn.model_selection import train_test_split

train,test = train_test_split(bankdata,test_size=0.3)

# checking na values 
train.isnull().sum();test.isnull().sum()

# Building a model on train data set 

train_model = sm.logit('y~age+job+marital+education+default+balance+housing+loan+contact+month+duration+campaign+previous+poutcome',data = train).fit()

#summary
train_model.summary()
train_pred = train_model.predict(train)

# Creating new column for storing predicted class of y

# filling all the cells with zeroes
train["train_pred"] = np.zeros(31647)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
train.loc[train_pred>0.5,"train_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(train['y'],train.train_pred)

confusion_matrix
accuracy_train = (27347+1031)/(31647) # 89.67
accuracy_train

# Prediction on Test data set

test_pred = train_model.predict(test)

# Creating new column for storing predicted class of y

# filling all the cells with zeroes
test["test_pred"] = np.zeros(13564)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
test.loc[test_pred>0.5,"test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test['y'],test.test_pred)

confusion_matrix
accuracy_test = (11741+400)/(13564) # 89.51
accuracy_test
