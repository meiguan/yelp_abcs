import warnings


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from sklearn.metrics import classification_report

from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_validate, GridSearchCV

from sklearn import preprocessing

# from sklearn import cross_validation
from sklearn import svm
from xgboost import XGBClassifier
from xgboost import plot_importance

from sklearn.feature_selection import f_classif, SelectKBest


import psycopg2


warnings.filterwarnings("ignore")


## DEFINE X and Y


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
Cs = [0.1, 1, 10]

select_Kbest = [10, 100, 'all']

params = {
    'svc__C': Cs,
    'svc__kernel': kernel_list,
    'svc__random_state': [0],
    'selectkbest__k': select_Kbest
}

classifier_pipeline = make_pipeline(SelectKBest(score_func=f_classif), svm.SVC())

gscv = GridSearchCV(classifier_pipeline,
                    [params],
                    cv=5,
                    scoring=('accuracy', 'precision', 'recall', 'roc_auc', 'f1'),
                    return_train_score=True,
                    refit=False,
                    verbose=1,
                    n_jobs=-1
                    )

gscv_model = gscv.fit(X, y)


# find the best performing model parameters
best_auc = gscv_model.cv_results_['mean_test_roc_auc'].max()
index_of_best_auc = gscv_model.cv_results_['mean_test_roc_auc'].argmax()

if best_auc != gscv_model.cv_results_['mean_test_roc_auc'][index_of_best_auc]:
    print "mismatch on the best AUC"

print "acc: ", gscv_model.cv_results_['mean_test_accuracy'][index_of_best_auc]
print "precision: ", gscv_model.cv_results_['mean_test_precision'][index_of_best_auc]
print "recall: ", gscv_model.cv_results_['mean_test_recall'][index_of_best_auc]
print "AUC: ", gscv_model.cv_results_['mean_test_roc_auc'][index_of_best_auc]
print "F1: ", gscv_model.cv_results_['mean_test_f1'][index_of_best_auc]
print ""
print gscv.estimator


