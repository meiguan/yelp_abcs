import numpy as np
from sklearn.ensemble import RandomForestClassifier
try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    try:
        from sklearn.grid_search import GridSearchCV
    except ImportError:
        try:
            from sklearn.cross_validation import GridSearchCV
        except ImportError:
            pass
from sklearn import preprocessing
import warnings

warnings.filterwarnings("ignore")


## DEFINE X and Y

# run model


# Grid search - random forest


# TAKES A LONG TIME
n_estimators = [int(x) for x in np.linspace(1, 300, 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# max_features = ["auto"]
# Maximum number of levels in tree
max_depth = [2, 4, 8]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# min_samples_split = [2]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# min_samples_leaf = [1]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# bootstrap = [True]
params = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
}

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

gscv = GridSearchCV(
    RandomForestClassifier(),
    params,
    cv=5,
    scoring=("accuracy", "precision", "recall", "roc_auc", "f1"),
    return_train_score=True,
    refit=False,
)
gscv_model = gscv.fit(X, y)

best_auc = gscv_model.cv_results_['mean_test_roc_auc'].max()
index_of_best_auc = gscv_model.cv_results_['mean_test_roc_auc'].argmax()

if best_auc != gscv_model.cv_results_['mean_test_roc_auc'][index_of_best_auc]:
    print("mismatch on the best AUC")

print("acc: ", gscv_model.cv_results_['mean_test_accuracy'][index_of_best_auc])
print("precision: ", gscv_model.cv_results_['mean_test_precision'][index_of_best_auc])
print("recall: ", gscv_model.cv_results_['mean_test_recall'][index_of_best_auc])
print("AUC: ", gscv_model.cv_results_['mean_test_roc_auc'][index_of_best_auc])
print("F1: ", gscv_model.cv_results_['mean_test_f1'][index_of_best_auc])
print("")
print(gscv.estimator)
