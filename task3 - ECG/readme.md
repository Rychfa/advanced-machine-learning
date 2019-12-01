Preprocessing (for each sample) consists of ignoring nans and normalize the signal with ecg.st.normalize. 
Then the rpeaks, filtered signal and time are taken using bioSPPY (300Hz).
First, the rr intervals in seconds between 2 consecutive rpeaks and the differences between 2 consecutive rr intervals are calculated.
The following features are extracted (total of 21):
- the bpm from rr intervals
- mean of rr intervals
- std of the rr intervals
- std of the rr intervals differences
- root square mean of the rr intervals differences
- proportion of the rr intervals differences greater than 0.2 seconds and 0.5 seconds
- the median, min, max of the:
   - Q amplitudes (directional change detection by running from rpeaks to the left), 
   - S amplitudes (directional change detection by running from rpeaks to the right) 
   - R amplitudes (directly taken from the rpeaks and the filtered signal)
- the mean and std of the qs interval in seconds
- the square root mean and std of the diffs of the qs durations
- the wavelet energy using the ecg.st.power_pectrum, ecg.st.band_power

The feature dataset is scaled using a minmax scaler. Then a grid search with a StratifiedShuffleSplit 10-fold is applied.
Best training score obtained using AdaBoostClassifier std: 0.012665 score: 0.79785 
``` python
AdaBoostClassifier(algorithm='SAMME.R',
                   base_estimator=DecisionTreeClassifier(min_samples_split=4),
                   learning_rate=0.061111111111111116,
                   n_estimators=35)
```
