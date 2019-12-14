import warnings

from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')


## DEFINE X and Y

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

loss = ['deviance', 'exponential']
learning_rate = [0.1, 1, 0.01, 0.001]
n_estimators = [100, 200, 300, 400]
criterion = ['friedman_mse']
max_depth = [3, 6, 9]
random_state = [0]
select_Kbest = [10, 50, 100, 200, 'all']

params = {
    'loss': loss,
    'learning_rate': learning_rate,
    'n_estimators': n_estimators,
    'criterion': criterion,
    'max_depth': max_depth,
    'random_state': random_state
}

gscv = GridSearchCV(GradientBoostingClassifier(),
                    [params],
                    cv=5,
                    scoring=('accuracy', 'precision', 'recall', 'roc_auc', 'f1'),
                    return_train_score=True,
                    refit=False,
                    verbose=1,
                    n_jobs=-1
                    )

gscv_model = gscv.fit(X, y)

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
