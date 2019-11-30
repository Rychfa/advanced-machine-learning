To achieve the best reslults in this task, the the training of the model has to take into account the class imbalance. 
The SVC classifier was used, and allows to cope with class imbalanced dataset by setting the class_weight parameter to 'balanced' 
By doing so, it adjusts the weights inversely proportional to class frequencies in the input data.
No particular preprocessing was needed, a part from using a robust scaler.

A pipeline was used over a StratifiedShuffleSplit of 10 splits to tune the hyperparameters. 
This allows having a random permutation of the cross validation while preserving the same proportion of each class. 
The pipeline contains:
- RobustScaler(quantile_range=(25.0, 75.0))
- SVC(C=1.0666666666666667, cache_size=1500,
                     class_weight='balanced', coef0=0.0,
                     decision_function_shape='ovo', degree=3, gamma='auto',
                     kernel='rbf', max_iter=-1, probability=False,
                     random_state=None, shrinking=True, tol=1e-12,
                     verbose=False)

Best training score achieved is 0.70222 with a standard deviation of 0.01993.
