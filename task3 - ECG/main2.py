import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from taskutils import perform_cv, write_results

# READ DATA
train_data_path = "X_train_features.csv"
label_data_path = "data/y_train.csv"
test_data_path = "X_test_features.csv"

X_train = pd.read_csv(train_data_path, header=None).values
y = pd.read_csv(label_data_path)['y'].values
X_test = pd.read_csv(test_data_path, header=None).values

# TRAIN
Cs = 10. ** np.arange(6, 8)
gammas = 10. ** np.arange(-3, 0)
parameters_svc = {
    'classifier__kernel': ['rbf'],
    'classifier__C': Cs,  # [2000, 3000, 4000, 5000, 6000, 7000],
    'classifier__coef0': [0.0],
    'classifier__decision_function_shape': ['ovr'],
    'classifier__cache_size': [1500],
    'classifier__gamma': ['auto', 0.4, 0.35, 0.3, 0.25, 0.2],
    'classifier__class_weight': ['balanced']
}

parameters_mlp = {
    'classifier__activation': ['tanh'],
    'classifier__alpha': np.linspace(10e-8, 10e-6, 20),  # 10. ** np.arange(-10, -6),
    'classifier__batch_size': ['auto'],
    'classifier__beta_1': [0.9],
    'classifier__beta_2': [0.999],
    'classifier__early_stopping': [False],
    'classifier__epsilon': [1e-12, 1e-08],
    'classifier__hidden_layer_sizes': [(15, 15), (15, 15, 15),
                                       (X_train.shape[1], X_train.shape[1]),
                                       (42, 42), (42, 42, 42)],
    # 'classifier__hidden_layer_sizes': [(len(X_train), len(X_train)), (len(X_train), len(X_train), len(X_train))],
    'classifier__learning_rate': ['constant'],
    'classifier__learning_rate_init': [0.007, 0.01, 0.02],  # 10. ** np.arange(-4, 0),
    'classifier__max_iter': [200],
    'classifier__momentum': [0.9],
    'classifier__n_iter_no_change': [10],
    'classifier__nesterovs_momentum': [True],
    'classifier__power_t': [0.5],
    'classifier__random_state': [1],
    'classifier__shuffle': [True],
    'classifier__solver': ['adam'],
    'classifier__tol': [0.0001],
    'classifier__validation_fraction': [0.1],
    'classifier__warm_start': [False]
}

parameters_knc = {
    'classifier__n_neighbors': np.arange(5, 50, 5),  # [2, 5, 10, 100],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__algorithm': ['kd_tree'],
    'classifier__p': [1, 2, 3, 4, 5],
    'classifier__metric': ['chebyshev', 'minkowski'],
    'classifier__leaf_size': np.arange(5, 30, 5)  # [10, 30, 60]
}

parameters_adaboost = {
    'classifier__n_estimators': np.arange(5, 50, 5),  # [2, 5, 10, 100],
    'classifier__learning_rate': np.linspace(0.05, 0.15, 10),
    'classifier__base_estimator__max_depth': [None],
    'classifier__base_estimator__min_samples_split': [4, 5, 6]
}

scaler = MinMaxScaler()

# std: 0.01639810995980653 // score: 0.7755859375
classifierMlp = MLPClassifier(activation='tanh', alpha=8.957894736842106e-06,
                              batch_size='auto', beta_1=0.9, beta_2=0.999,
                              early_stopping=False, epsilon=1e-08,
                              hidden_layer_sizes=(15, 15),
                              learning_rate='constant',
                              learning_rate_init=0.007, max_iter=200,
                              momentum=0.9, n_iter_no_change=10,
                              nesterovs_momentum=True, power_t=0.5,
                              random_state=1, shuffle=True, solver='adam',
                              tol=0.0001, validation_fraction=0.1,
                              verbose=False, warm_start=False)

# std: 0.019161539912449788 // score: 0.7685546875
classifierSvc = SVC(C=7000, cache_size=1500, class_weight='balanced',
                    coef0=0.0, decision_function_shape='ovr', degree=3,
                    gamma=0.3, kernel='rbf', max_iter=-1, probability=True,
                    random_state=None, shrinking=True, tol=0.001,
                    verbose=False)

classifierKnc = KNeighborsClassifier(algorithm='kd_tree', leaf_size=5,
                                     metric='minkowski', metric_params=None,
                                     n_jobs=None, n_neighbors=10, p=1,
                                     weights='distance')  # MLPClassifier()  # SVC() #

# std: 0.012665228778807403 // score: 0.7978515625
classifierAda = AdaBoostClassifier(algorithm='SAMME.R',
                                   base_estimator=DecisionTreeClassifier(class_weight=None,
                                                                         criterion='gini',
                                                                         max_depth=None,
                                                                         max_features=None,
                                                                         max_leaf_nodes=None,
                                                                         min_impurity_decrease=0.0,
                                                                         min_impurity_split=None,
                                                                         min_samples_leaf=1,
                                                                         min_samples_split=4,
                                                                         min_weight_fraction_leaf=0.0,
                                                                         presort=False,
                                                                         random_state=None,
                                                                         splitter='best'),
                                   learning_rate=0.061111111111111116,
                                   n_estimators=35, random_state=None)
estimators = [('ada', classifierAda), ('mlp', classifierMlp), ('svc', classifierSvc)]

classifier = VotingClassifier(estimators=estimators, voting='soft', weights=[8, 3, 1])
pipe_line = Pipeline([
    ('scaler', scaler),
    ('classifier', classifier)
])

n_splits = 10
cross_validation_model = StratifiedShuffleSplit(n_splits=n_splits)
model = perform_cv(pipe_line, X_train, y, {}, 'f1_micro', cross_validation_model, 10, 3)

# TEST
y_pred = model.predict(X_test)
write_results(classifier, range(len(X_test)), y_pred)
