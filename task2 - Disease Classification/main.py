import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from taskutils import read_input, perform_cv, write_results


def analyse_data(data):
    nans = np.isnan(data)
    nan_count = sum(nans)
    mins = np.min(data, axis=1)
    maxs = np.max(data, axis=1)
    means = np.mean(data, axis=1)
    mids = np.median(data, axis=1)
    stds = np.std(data, axis=1)

    for i in range(len(data)):
        print("datum {} - class {} --> min: {} mean: {} mid: {} max: {} stds: {} | nans: {}".format(i, y[i], mins[i],
                                                                                                    means[i], mids[i],
                                                                                                    maxs[i], stds[i],
                                                                                                    sum(nans[i])))


########################################################################################################################
########################################################################################################################

# READ DATA
train_data_path = "data/X_train.csv"
label_data_path = "data/y_train.csv"
test_data_path = "data/X_test.csv"

X_train = read_input(train_data_path).values[:, 1:]
y = pd.read_csv(label_data_path)['y'].values
X_test = read_input(test_data_path).values[:, 1:]

# PREPARE MODEL
scaler = RobustScaler()
feature_selector = SelectKBest()
classifier = SVC(class_weight='balanced')
pipe_line = Pipeline([
    ('scaler', scaler),
    ('classifier', classifier)
])

# CROSS VALIDATION
Cs = 10. ** np.arange(-1, 4)
tols = 10. ** np.arange(-12, -9)
parameters_svc = {
    'classifier__kernel': ['rbf'],
    'classifier__C': [1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4], #np.linspace(1.0, 1.3, 10),  # range(1000, 10000, 1000),
    'classifier__cache_size': [1500],
    'classifier__gamma': ['auto'],
    'classifier__tol': [10**-13],
    'classifier__decision_function_shape': ['ovo']
}

# classifier = MLPClassifier()
# parameters_mlpc = {
#     'classifier__hidden_layer_sizes': [(50, 50, 50), (500, 500, 500), (1000, 500, 500, 200)],
#     'classifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
#     'classifier__alpha': [10**-8, 10**-4, 10**0, 5],
#     'classifier__tol': [10**-10]
# }

# TRAINING PHASE
n_splits = 10
cross_validation_model = StratifiedShuffleSplit(n_splits=n_splits)
model = perform_cv(pipe_line, X_train, y, parameters_svc, 'balanced_accuracy', cross_validation_model, 6, 10)


# TESTING PHASE
y_pred = model.predict(X_test)
write_results(classifier, range(len(X_test)), y_pred)
