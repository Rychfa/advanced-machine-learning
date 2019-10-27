To achieve the best reslults in this task, preprocessing the data before training the model for the final prediction were necessary.
First, the data set was missing some values, and all of these were imputed using the mean method.
Second, the data set contained some outliers which adds some noise to the model and make it less robust. 
To overcome this issue, a outlier detection model was used to remove all the sample that was detected as such. 
LocalOutlierFactor was used with n_neighbors=40 seems to give the best results.

After the data was imputed and outliers removed, a pipeline was used over a StratifiedKFold=10 to tune the hyperparameters. The pipeline contains (in order):
- a RobustScaler with quantile range (25.0, 75.0), 
- a SelectKBest with k=174 (feature selection) 
- and then an SVR regressor with C=36, epsilon=0.01, gamma='auto', kernel='rbf', tol=1e-12

During testing phase, same preprocessing was used (naturally without removing the outlier detection step)
Finally order to improve the results, multiple models can be added to the SVR used here with a voting ensemble and tune the weights parameters of each model to keep the highest possible score.
