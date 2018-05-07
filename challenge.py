import pandas as pd
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, BayesianRidge, HuberRegressor

classification_data_train = pd.read_csv('class_train.csv')
regression_data_train = pd.read_csv('regre_train.csv')
classification_data_test = pd.read_csv('class_test.csv')
regression_data_test = pd.read_csv('regre_test.csv')

class_array_train = classification_data_train.values
regre_array_train = regression_data_train.values
class_array_test = classification_data_test.values
regre_array_test = regression_data_test.values

validation_size = 0.2
seed = 7
scoring = 'accuracy'
X_len = len(class_array_train[0])
X = class_array_train[:,0:X_len - 1]
Y = class_array_train[:,X_len - 1]

X_train, X_validate, Y_train, Y_validate = model_selection.train_test_split(X,
                                Y, test_size=validation_size, random_state=seed)

models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('DecisionTree', DecisionTreeClassifier()))
models.append(('MLP', MLPClassifier()))
models.append(('NaiveBayes', GaussianNB()))
models.append(('AdaBoost', AdaBoostClassifier()))
models.append(('RandomForest', RandomForestClassifier()))
models.append(('SVM', SVC()))


results = []
names = []
print("CLASSIFICATION")
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


X_len = len(regre_array_train[0])
X = regre_array_train[:,0:X_len - 1]
Y = regre_array_train[:,X_len - 1]
scoring = 'r2'


X_train, X_validate, Y_train, Y_validate = model_selection.train_test_split(X,
                                Y, test_size=validation_size, random_state=seed)

print("REGRESSION")
models = []
results = []
names = []
models.append(('LogisticRegression', LogisticRegression()))
models.append(('LinearRegression', LinearRegression()))
models.append(('ElasticNet', ElasticNet()))
models.append(('BayesianRidge', BayesianRidge()))
models.append(('HuberRegressor', HuberRegressor()))


for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#
# kf = model_selection.KFold(n_splits=2)
# kf.get_n_splits(class_array_train)
#
# print(kf)
#
# model_selection.KFold(n_splits=2, random_state=None, shuffle=False)
#
# for train_index, test_index in kf.split(class_array_train):
#     print("TRAIN: ", train_index, "TEST: ", test_index)
#     #X_train, X_test
