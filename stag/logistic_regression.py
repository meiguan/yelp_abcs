import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn import preprocessing


RANDOM_SEED = 666


## DEFINE X and Y

scaler = preprocessing.StandardScaler()
selector = SelectKBest(f_classif)
logistic = LogisticRegression()

pipe = Pipeline(steps=[('scaler', scaler), ('selector', selector), ('logistic', logistic)])

params = {
    'selector__k': [10, 50, 100, 200, 'all'],
    'logistic__penalty': ['l2', 'l1'],
    'logistic__C': [1e-2, 1e-1, 1, 1e1, 1e2]
}
gscv = GridSearchCV(pipe, params, cv=5,
                          scoring=('accuracy', 'precision', 'recall', 'roc_auc', 'f1'),
                          return_train_score=True,
                          refit='roc_auc')
gscv_model = gscv.fit(X, y)

print "Mean Accuracy: \t", gscv_model.cv_results_['mean_test_accuracy'].mean()
print "Mean Precision:", gscv_model.cv_results_['mean_test_precision'].mean()
print "Mean Recall: \t", gscv_model.cv_results_['mean_test_recall'].mean()
print "Mean AUC: \t", gscv_model.cv_results_['mean_test_roc_auc'].mean()
print "Mean F1: \t", gscv_model.cv_results_['mean_test_f1'].mean()
print

print(gscv_model.best_params_)
print

for score in gscv_model.cv_results_:
    if 'mean_test' in score:
        print score + ': ', gscv_model.cv_results_[score][gscv_model.best_index_]

best_auc = gscv_model.cv_results_['mean_test_roc_auc'].max()
index_of_best_auc = gscv_model.cv_results_['mean_test_roc_auc'].argmax()

if best_auc != gscv_model.cv_results_['mean_test_roc_auc'][index_of_best_auc]:
    print("mismatch on the best AUC")

print("Values for the model with the highest AUC:")
print("acc: ", gscv_model.cv_results_['mean_test_accuracy'][index_of_best_auc])
print("precision: ", gscv_model.cv_results_['mean_test_precision'][index_of_best_auc])
print("recall: ", gscv_model.cv_results_['mean_test_recall'][index_of_best_auc])
print("AUC: ", gscv_model.cv_results_['mean_test_roc_auc'][index_of_best_auc])
print("F1: ", gscv_model.cv_results_['mean_test_f1'][index_of_best_auc])
print("")
print(gscv.estimator)

def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.title('Top 20 Coefficients of Logistic Regression Model')
    plt.ylabel('Coefficients')
    plt.show()


best_lr = gscv_model.best_estimator_.steps[2][1]
best_cols = filtered_df.columns[gscv_model.best_estimator_.steps[1][1].get_support(indices=True)]
plot_coefficients(best_lr, best_cols)

feat_coef = sorted(zip(best_cols, best_lr.coef_[0]), key=lambda x: x[1], reverse=True)

# Pretty print
# col_width = max(len(word) for word in best_cols) + 2  # padding
# for feat, coef in feat_coef:
#     print feat.ljust(col_width) + str(coef)

for feat, coef in feat_coef:
    print feat , ',' , str(coef)
