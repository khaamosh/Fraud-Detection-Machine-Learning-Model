import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# from sklearn.svm import SVC
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

train_transaction = pd.read_csv('ieee-fraud-detection/train_transaction.csv')
test_transaction = pd.read_csv('ieee-fraud-detection/test_transaction.csv')

sample = pd.read_csv('ieee-fraud-detection/sample_submission.csv')

print(train_transaction.shape, test_transaction.shape)
print(sample.shape)

print(train_transaction.head(10))

# Dropping columns with more than 50% missing values.
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

# random forest

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

# svm
# no result
params_svm_temp1 = {
    'kernel': 'rgb',
    'gamma': 0.001,
}
# no result
params_svm_temp2 = {
    'kernel': 'rgb',
    'random_state': 1,
    'probability': True,
    'gamma': 0.01,
}
# no result
params_svm_temp3 = {
    'kernel': 'linear',
    'random_state': 5,
    'probability': True,
    'gamma': 1000,
}
# no result
params_svm = {
    'kernel': 'linear',
    'random_state': 1,
    'probability': True,
    'gamma': 100,
    'random_state': 108,
    'decision_function_shape': 'ovo'
}
clf_svm = SVC(**params_svm)
clf_svm.fit(X_train, Y_train)
Y_pred_svm = clf_svm.predict_proba(X_test)
sample['isFraud'] = Y_pred_svm[:, 1]
sample.to_csv('submission_svm.csv', index=False)

# lgbm
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
sample.to_csv('submission_lgbm.csv', index=False)
