import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, VotingRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, mutual_info_regression, chi2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, DotProduct, RBF, Exponentiation
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, TheilSenRegressor, HuberRegressor, BayesianRidge
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVR, LinearSVC, SVC

from taskutils import read_input, perform_cv, write_results

# READ DATA
mri_train_data_path = "data/X_train.csv"
mri_label_data_path = "data/y_train.csv"
mri_test_data_path = "data/X_test.csv"

mri_train_data = read_input(mri_train_data_path).values[:, 1:]
mri_label_data = pd.read_csv(mri_label_data_path)['y'].values
mri_test_data = read_input(mri_test_data_path).values[:, 1:]

# # REMOVE WEIRD LONELY POINTS (FROM LABELS)
# labels = {}
# for y in mri_label_data:
#     label_key = "{}".format(y)
#     labels[label_key] = 1 if label_key not in labels else labels[label_key] + 1
#
# occurrence_label = 1
# to_exclude = []
# for idx, key in enumerate(labels):
#     if labels[key] <= occurrence_label:
#         to_exclude.append(idx)
#         print("id {} only {} occurrence for age {}".format(idx, occurrence_label, key))
#
# mri_label_data = mri_label_data[~np.in1d(range(len(mri_label_data)), to_exclude)]
# mri_train_data = np.delete(mri_train_data, to_exclude, 0)

# IMPUTE NANs
imputer = SimpleImputer(strategy='mean')
mri_train_data = imputer.fit_transform(mri_train_data)

# PREPARE MODEL
scaler = RobustScaler()
feature_selector = SelectKBest()
# classifier = BayesianRidge(n_iter=1000, tol=0.00001)
classifier = SVR()
# svr_opt = SVR(C=36, cache_size=1500, coef0=0.0, degree=3, epsilon=0.01,
#                      gamma='auto', kernel='rbf', max_iter=-1, shrinking=True,
#                      tol=1e-12, verbose=False)
#
# gpr_kernel = Exponentiation(DotProduct(sigma_0=0.1), 1.0) + WhiteKernel(noise_level=5.0)
# gpr_opt = GaussianProcessRegressor(alpha=0.0003, copy_X_train=True, kernel=gpr_kernel,
#                                           n_restarts_optimizer=0,
#                                           normalize_y=False,
#                                           optimizer='fmin_l_bfgs_b',
#                                           random_state=0)

# classifier = GaussianProcessRegressor(kernel=gpr_kernel, random_state=0)
# classifier = VotingRegressor([('svr', svr_opt), ('gpr', gpr_opt)])

pipe_line = Pipeline(
    [
        ('scaler', scaler),
        ('feature_selector', feature_selector),
        ('classifier', classifier)
    ]
)

# CROSS VALIDATION
# Cs = 10. ** np.arange(-1, 4)
Cs = range(30, 40, 1)  # np.arange(20, 500, 10)
# Cs = range(50, 500, 50) #np.arange(20, 500, 10)
# gammas = 10. ** np.arange(-8, -6)
tols = 10. ** np.arange(-12, -9)
epsilons = 10.0 ** np.arange(-6, -1)

parameters_svr = {
    'classifier__kernel': ['rbf'],
    # 'classifier__C': range(40, 60, 2), #range(1000, 7000, 2000), #[40, 50, 60],
    'classifier__C': Cs,  # range(1000, 10000, 1000),
    'classifier__cache_size': [1500],
    'classifier__gamma': ['auto'],
    # 'classifier__gamma': ['auto'],
    'classifier__tol': tols,
    'classifier__epsilon': np.linspace(0.005, 0.015, 20),  # epsilons,
    'feature_selector__score_func': [f_regression],
    'feature_selector__k': [174],  # range(170, 175, 1)#,
    # 'scaler__quantile_range': [(25.0, 75.0), (30.0, 70.0)]
    # 'feature_selector__estimator__C': 10. ** np.arange(0, 4)
    # 'feature_selector__estimator__solver': ['liblinear']
}

parameters_bayes = {
    'classifier__alpha_1': 10. ** np.arange(-20, -8),
    'classifier__alpha_2': 10. ** np.arange(-3, -1),#[3.0, 5.0, 10.0],
    'classifier__lambda_1': 10. ** np.arange(-3, -1),
    'classifier__lambda_2': 10. ** np.arange(-20, -8),
    'feature_selector__score_func': [f_regression],
    'feature_selector__k': [175]  # range(170, 175, 1)#,
}

parameters_ada = {
    'classifier__n_estimators': [80, 150],
    'classifier__learning_rate': np.linspace(1.5, 3.5, 50),#[3.0, 5.0, 10.0],
    'classifier__loss': ['linear', 'exponential'],
    'feature_selector__score_func': [f_regression],
    'feature_selector__k': [175]  # range(170, 175, 1)#,
}

parameters_gpr = {
    'classifier__alpha': [0.0001, 0.0002, 0.0003, 0.0005],
    'feature_selector__score_func': [f_regression],
    'feature_selector__k': [175]
}

# parameters_voting = {
#     'classifier__weights': [(1, 0)],
#     'feature_selector__score_func': [f_regression],
#     'feature_selector__k': [175]
# }


best_model = None
best_neighbors = None

for neighbors in [40, 45, 50]:

    # OUTLIER DETECTION
    outlier_detector = LocalOutlierFactor(n_neighbors=neighbors, contamination='auto')
    outliers = outlier_detector.fit_predict(mri_train_data, mri_label_data)
    print("{} neighbors -> {} outliers".format(outlier_detector.n_neighbors, sum(outliers == -1)))
    mri_train_data_X = mri_train_data[outliers == 1, :]
    mri_label_data_y = mri_label_data[outliers == 1]

    # TRAINING PHASE
    cross_validation_model = StratifiedKFold(n_splits=12)
    model = perform_cv(pipe_line, mri_train_data_X, mri_label_data_y, parameters_svr, 'r2', cross_validation_model, 8, 1)

    # # TESTING PHASE
    # mri_test_data = imputer.transform(mri_test_data)  # OR FIT TRANSFORM ?
    # y_pred = model.predict(mri_test_data)
    # write_results(classifier, range(len(mri_test_data)), y_pred)

    # UPDATE BEST
    should_update = (best_model is None) or (model.best_score_ > best_model.best_score_)
    if should_update:
        best_model = model
        best_neighbors = neighbors
    # best_model = model if should_update is True else best_model
    # best_neighbors = neighbors if should_update is True else best_neighbors

    print("{} :  best_model {} vs model {}".format(should_update, best_model.best_score_, model.best_score_))

    # print("{} neighbors -> {} outliers".format(outlier_detector.n_neighbors, sum(outliers == -1)))
    print("\n###########################################################")
    print("###########################################################\n")

# TESTING PHASE
mri_test_data = imputer.transform(mri_test_data)  # OR FIT TRANSFORM ?
y_pred = best_model.predict(mri_test_data)
write_results(classifier, range(len(mri_test_data)), y_pred)

print("*********************************")
print("BEST MODEL --> {}".format(classifier.__class__.__name__))
print("*********************************")
print("best neighbors: {}".format(best_neighbors))
print("best estimator: {}".format(best_model.best_estimator_))
print("std: {}".format(best_model.cv_results_['std_test_score'][best_model.best_index_]))
print("score: {}".format(best_model.best_score_))
print("params: {}".format(best_model.best_params_))
