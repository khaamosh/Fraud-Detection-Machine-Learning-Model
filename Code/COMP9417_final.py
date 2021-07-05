#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load main.py
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV


# In[2]:


#for printing the descision Tree
from sklearn.tree.export import export_text

# from sklearn.svm import SVC
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


# In[3]:


#using BernoulliNB
from sklearn.naive_bayes import  BernoulliNB

#the file variables to download the information
train_transaction = pd.read_csv('ieee-fraud-detection/train_transaction.csv')
test_transaction = pd.read_csv('ieee-fraud-detection/test_transaction.csv')

sample = pd.read_csv('ieee-fraud-detection/sample_submission.csv')

print(train_transaction.shape, test_transaction.shape)
print(sample.shape)

print(train_transaction.head(10))

################################################################################
#This section corresponds to the EDA for the data and the graphs which are created are as follows :
#EDA code for the main file is as follows :


# In[4]:


#get the basics for the train transaction
train_transaction.head()
# checking for the nan values since we have huge amount of features, hence putting the transaction data into a dataframe
train_null_dataframe = train_transaction.isnull().sum()
#The distribution of the transaction of the data.

# Since values were large hence for plotting they were normalized using the log values for the cases.
train_transaction['TransactionAmt'].apply(np.log).plot(kind='hist',bins=80,figsize=(15,5),title="Distribution of Transactions")
plt.show()


#checking the data for the valid transaction
Valid_transaction = train_transaction[train_transaction['isFraud']==0]
Valid_transaction.head()


#Getting the distribution of the amount for fraud transactions

#this distribution would help in checking for the above average amount transactions which are fraud.
train_transaction[train_transaction['isFraud']==1]['TransactionAmt'].apply(np.log).plot(kind='hist',bins=100,figsize=(15,5),title="Transaction Amount for Fraud Transaction")
plt.show()


# c_cols = [c for c in train_transaction if c[0] == 'C']
# train_transaction[c_cols].head()

#
# #We used pairplot to check if intervariable dependence is there and also for individual analysis
# sns.pairplot(train_transaction[train_transaction['isFraud']==1], vars = c_cols, dropna = True, hue = 'isFraud')

###################################################################################################################


# In[5]:


#Data Pre-Processing is done here for the dataset

# Dropping columns with more than 50% missing values, checking the information given from the graphs

columns_to_drop = []
num_of_rows = train_transaction.shape[0]
for i in train_transaction.columns:
    count_of_null_values = train_transaction[i].isna().sum()
    if (count_of_null_values >= num_of_rows/2):
        columns_to_drop.append(i)
del num_of_rows

# droping columns from both train and test dataset
train_transaction.drop(columns_to_drop, axis=1, inplace=True)
test_transaction.drop(columns_to_drop, axis=1, inplace=True)

# # categorical data
object_columns = train_transaction.select_dtypes(include=object).columns
print("Number of categorical columns: {}".format(len(object_columns)))

for i in object_columns:
    print("Column Name : {}".format(i))
    print(
        "-------------> No of missing values: {}".format(train_transaction[i].isna().sum()))
    print(
        "-------------> Unique values: {}".format(train_transaction[i].unique()))


# filling null values with mode
for i in object_columns:
    train_transaction[i].fillna(train_transaction[i].mode()[0], inplace=True)
    test_transaction[i].fillna(test_transaction[i].mode()[0], inplace=True)

cat_num_features = ['addr1', 'addr2', 'card1', 'card2', 'card3', 'card5']


# In[6]:


# Filling the missing values with mode.
for i in cat_num_features:
    train_transaction[i].fillna(train_transaction[i].mode()[0], inplace=True)
    test_transaction[i].fillna(test_transaction[i].mode()[0], inplace=True)
del cat_num_features

# numeric columns
all_numeric_columns = train_transaction.select_dtypes(
    include=np.number).columns
numeric_missing = []
for i in all_numeric_columns:
    missing = train_transaction[i].isna().sum()
    if(missing > 0):
        numeric_missing.append(i)
del all_numeric_columns
print(len(numeric_missing))


# Filling the missing values with median.
for i in numeric_missing:
    train_transaction[i].fillna(train_transaction[i].median(), inplace=True)
    test_transaction[i].fillna(test_transaction[i].median(), inplace=True)

print(train_transaction.isna().any().sum(),
      test_transaction.isna().any().sum())
del numeric_missing

# object_columns labelencoding
for f in object_columns:
    lbl = LabelEncoder()
    lbl.fit(list(train_transaction[f].values) +
            list(test_transaction[f].values))
    train_transaction[f] = lbl.transform(list(train_transaction[f].values))
    test_transaction[f] = lbl.transform(list(test_transaction[f].values))

print(len(train_transaction.select_dtypes(exclude=np.number).sum()))

# # saving clean data
# train_transaction.to_csv("clean_train_transaction.csv", sep=',', index=False)
# test_transaction.to_csv("clean_test_transaction.csv", sep=',', index=False)
# # reading clean data
# train_transaction = pd.read_csv('clean_train_transaction.csv')
# test_transaction = pd.read_csv('clean_test_transaction.csv')

Y_train = train_transaction['isFraud']
X_train = train_transaction.drop(['isFraud'], axis=1)
X_test = test_transaction

# checking and removing left over(if any) garbage/infinity values

X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.fillna(999, inplace=True)

'''
Check for feature selection done here, this process is commented out since we had picked the correct features from the given list.

Feature selection correction done here

percent = 25 # percent of dataset to use for feature selection

x_train_train, x_train_valid, y_train_train, y_train_valid = train_test_split(X_sm, y_sm, test_size=1.0-percent/100.0, random_state=42)


feature_selector = RFECV(BernoulliNB(), step=15, scoring='roc_auc', cv=5, verbose=1,n_jobs=3)
feature_selector.fit(x_train_train, y_train_train)
print('Features selected:', feature_selector.n_features_)


selected_features = [f for f in x_train_train.columns[feature_selector.ranking_ == 1]]

'''

#################################################################################
#SMOTE and class imbalance for the dataaset

#Graph which shows classes are balanced

#implementing the code for the BNB model and implementing class imbalance here
labels = ['Valid','Fraud']

#graph that shows imbalance in the class
plt.bar(labels,Y_train.value_counts())
plt.show()


#smote implemented here
smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_sample(X_train, Y_train)

labels = ['Valid','Fraud']

#graph showing balanced classes
plt.bar(labels,y_sm.value_counts())
plt.show()

################################################################################


# In[7]:


##############################################################################
#implementation of the descision Tree algorithm 
#Score :  

DTree = tree.DecisionTreeClassifier(random_state=0, criterion='entropy',max_depth=8,splitter='best', min_samples_split=30)
DTree = DTree.fit(X_train,Y_train)

pred = DTree.predict(X_test)
sample['isFraud'] = pred
sample.to_csv('submission_dt.csv', index=False)

'''
Priting the DTree here for analysis
r = export_text(DTree)
print(r)
'''
##############################################################################


# In[10]:


##############################################################################
#Implementation of the Bernoulli Algorithm
#Score : 74.4

#BernoulliNB model is here
#The hyper paramters used for tuning the model.

param_grid = {
                "alpha":[0.001,0.01,0.1,1],
    "fit_prior":[True]}                 

grid_bnb = GridSearchCV(BernoulliNB(),param_grid,cv=10,return_train_score=True)
model = grid_bnb.fit(X_sm,y_sm)

pred = grid_bnb.predict(X_test)

sample['isFraud'] = pred
sample.to_csv('submission_bnb_after_grid_after_k.csv', index=False)

##############################################################################

#Implementation of the KNN 


##############################################################################


# In[19]:


#Print the value post grid search
grid_bnb.best_estimator_


# In[11]:


#############################################################################

# Implementation of random forest

params_rf_temp1 = {
    'n_estimators': 100
}

params_rf_temp2 = {
    'n_estimators': 500,
    'random_state': 10,
    'max_depth': 20
}

params_rf_temp3 = {
    'n_estimators': 1000,
    'random_state': 200,
    'bootstrap': False,
    'max_depth': 5
}

params_rf = {
    'n_estimators': 1000,
    'random_state': 121,
    'min_samples_split': 2,
    'bootstrap': False,
    'max_depth': 5
}

clf_rf = RandomForestClassifier(**params_rf)
clf_rf.fit(X_train, Y_train)
Y_pred_rf = clf_rf.predict_proba(X_test)
sample['isFraud'] = Y_pred_rf[:, 1]
sample.to_csv('submission_rf.csv', index=False)


##############################################################################


# In[12]:


#############################################################################
# lgbm Implemented here
# Score= .8427
params_lgbm_temp1 = {
    'objective': 'binary',
    'n_estimators': 300,
    'learning_rate': 0.1,
    'subsample': 0.8
}
# Score= .8306
params_lgbm_temp2 = {
    'objective': 'binary',
    'n_estimators': 200,
    'learning_rate': 0.1,
}
#Score= .8446
params_lgbm_temp3 = {
    'objective': 'binary',
    'n_estimators': 300,
    'learning_rate': 0.1,
}
# Score=.874
params_lgbm_temp4 = {
    'objective': 'binary',
    'n_estimators': 600,
    'learning_rate': 0.1
}
#Score= .8666
params_lgbm_temp5 = {
    'objective': 'binary',
    'n_estimators': 500,
    'learning_rate': 0.1
}
#Score= .8911
params_lgbm_temp6 = {
    'objective': 'binary',
    'n_estimators': 500,
    'learning_rate': 0.1,
    'num_leaves': 50,
    'max_depth': 7,
    'subsample': 0.9,
    'colsample_bytree': 0.9
}
# Score= .90109
params_lgbm_temp7 = {
    'objective': 'binary',
    'n_estimators': 600,
    'learning_rate': 0.1,
    'num_leaves': 50,
    'max_depth': 7,
    'subsample': 0.9,
    'colsample_bytree': 0.9
}
# Score= .920670
params_lgbm = {
    'objective': 'binary',
    'n_estimators': 700,
    'learning_rate': 0.1,
    'num_leaves': 50,
    'max_depth': 7,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'random_state': 108
}

clf_lgbm = LGBMClassifier(**params_lgbm)
clf_lgbm.fit(X_train, Y_train)

Y_pred_lgbm = clf_lgbm.predict_proba(X_test)
sample['isFraud'] = Y_pred_lgbm[:, 1]
sample.to_csv('final_model.csv', index=False)

##############################################################################


# In[ ]:




