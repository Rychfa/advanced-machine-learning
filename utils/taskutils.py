import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

def read_input(csv_file_path):
    return pd.read_csv(csv_file_path)#.values[:, 1:]


def write_results(classifier, test_ids, y_pred):
    res_pred = pd.DataFrame(columns=['id', 'y'])
    res_pred['id'] = test_ids
    res_pred['y'] = y_pred
    res_pred.to_csv(
        "data/res/{}-{}.csv".format(classifier.__class__.__name__, dt.datetime.now().strftime('%Y.%m.%dT%H.%M.%S')),
        index=False)


# def perform_cv(classifier, X, y, grid_params, score):
#     k_cv = 5
#     cross_validation_model = StratifiedShuffleSplit(n_splits=k_cv)
#     grid_search = GridSearchCV(classifier, grid_params, score, cv=cross_validation_model, n_jobs=2, verbose=30)
#     grid_search.fit(X, y)
#     print("*********************************")
#     print("CLASSIFIER --> {}".format(classifier.__class__.__name__))
#     print("*********************************")
#     # print(grid_search.cv_results_)
#     # print("*********************************")
#     print("best estimator: {}".format(grid_search.best_estimator_))
#     print("std: {}".format(grid_search.cv_results_['std_test_score'][grid_search.best_index_]))
#     print("score: {}".format(grid_search.best_score_))
#     print("params: {}".format(grid_search.best_params_))
#
#     print("*********************************")
#     minstd_estimator_id = np.argmin(grid_search.cv_results_['std_test_score'])
#     std = grid_search.cv_results_['std_test_score'][minstd_estimator_id]
#     score = grid_search.cv_results_['mean_test_score'][minstd_estimator_id]
#     params = grid_search.cv_results_['params'][minstd_estimator_id]
#     print("min std estimator id: {}".format(minstd_estimator_id))
#     print("std: {}".format(std))
#     print("score: {}".format(score))
#     print("params: {}".format(params))
#     return grid_search


def perform_cv(classifier, X, y, grid_params, score, cross_validation_model, number_jobs, verbose=0):
    grid_search = GridSearchCV(classifier, grid_params, score, cv=cross_validation_model, n_jobs=number_jobs, verbose=verbose)
    grid_search.fit(X, y)
    print("*********************************")
    print("CLASSIFIER --> {}".format(classifier.__class__.__name__))
    print("*********************************")
    print("best estimator: {}".format(grid_search.best_estimator_))
    print("std: {}".format(grid_search.cv_results_['std_test_score'][grid_search.best_index_]))
    print("score: {}".format(grid_search.best_score_))
    print("params: {}".format(grid_search.best_params_))

    print("*********************************")
    minstd_estimator_id = np.argmin(grid_search.cv_results_['std_test_score'])
    std = grid_search.cv_results_['std_test_score'][minstd_estimator_id]
    score = grid_search.cv_results_['mean_test_score'][minstd_estimator_id]
    params = grid_search.cv_results_['params'][minstd_estimator_id]
    print("min std estimator id: {}".format(minstd_estimator_id))
    print("std: {}".format(std))
    print("score: {}".format(score))
    print("params: {}".format(params))
    return grid_search

def plot_image(image, cmap=None):
    plt.clf()
    plt.imshow(image, cmap=cmap)
    plt.show()

def plot_image2(image, points):
    plt.clf()
    plt.imshow(image)
    plt.plot(points[:, 1], points[:, 0], '+r', markersize=30)
    plt.show()


def plot_image3(image, points, edges, fig):
    # plt.clf()
    # fig = plt.figure()
    fig.clf()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Original + corner detection')
    plt.imshow(image)
    plt.plot(points[:, 1], points[:, 0], '+r', markersize=30)

    a = fig.add_subplot(1, 2, 2)
    a.set_title('Edge detection')
    plt.imshow(edges)
    plt.show()


def plot_image4(image, points, edges, points2, fig):
    # plt.clf()
    # fig = plt.figure()
    fig.clf()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Original + corner detection')
    plt.imshow(image)
    plt.plot(points[:, 1], points[:, 0], '+r', markersize=20)

    a = fig.add_subplot(1, 2, 2)
    a.set_title('Edge detection')
    plt.imshow(edges)
    plt.plot(points2[:, 1], points2[:, 0], '+r', markersize=20)
    plt.show()


def plot_t5(eeg1, eeg2, emg, fig):
    # plt.clf()
    # fig = plt.figure()
    fig.clf()
    a = fig.add_subplot(1, 3, 1)
    a.set_title('eeg1')
    plt.plot(eeg1)

    a = fig.add_subplot(1, 3, 2)
    a.set_title('eeg2')
    plt.plot(eeg2)
    plt.show()

    a = fig.add_subplot(1, 3, 3)
    a.set_title('emg')
    plt.plot(emg)
    plt.show()
