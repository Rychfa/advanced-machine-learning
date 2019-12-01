import numpy as np
from scipy.interpolate import interp1d
from biosppy.signals import ecg


def find_s_point(ecg, R_peaks):
    num_peak = R_peaks.shape[0]
    S_point = list()
    for index in range(num_peak):
        i = R_peaks[index]
        cnt = i
        if cnt + 1 >= ecg.shape[0]:
            break
        while ecg[cnt] > ecg[cnt + 1]:
            cnt += 1
            if cnt >= ecg.shape[0]:
                break
        S_point.append(cnt)
    return np.asarray(S_point)


def find_q_point(ecg, R_peaks):
    num_peak = R_peaks.shape[0]
    Q_point = list()
    for index in range(num_peak):
        i = R_peaks[index]
        cnt = i
        if cnt - 1 < 0:
            break
        while ecg[cnt] > ecg[cnt - 1]:
            cnt -= 1
            if cnt < 0:
                break
        Q_point.append(cnt)
    return np.asarray(Q_point)


def extract_rr_intervals(rpeaks_time):
    rr_intervals = rpeaks_time[1:] - rpeaks_time[0:-1]
    return rr_intervals


def extract_rr_diffs(rr_intervals):
    rr_diffs = rr_intervals[1:] - rr_intervals[0:-1]
    return rr_diffs


def extract_rr_interval(rpeaks_time):
    rr_intervals = rpeaks_time[1:] - rpeaks_time[0:-1]
    rr_interval = np.median(rr_intervals)
    rr_var = np.var(rr_intervals)
    rr_min = np.min(rr_intervals)
    rr_max = np.max(rr_intervals)
    return [rr_interval, rr_min, rr_max, rr_var]


def extract_frequency(rpeaks, rr_intervals, signal_length, sampling_rate):
    RR_x = rpeaks[1:]  # Remove the first entry, because first interval is assigned to the second beat.
    RR_y = rr_intervals  # Y-values are equal to interval lengths
    # Create evenly spaced timeline starting at the second peak, its endpoint and length equal to position of last peak
    RR_x_new = np.linspace(RR_x[0], RR_x[-1], RR_x[-1])
    f = interp1d(RR_x, RR_y, kind='cubic')
    # Set variables
    fft_split = int(signal_length / 2) - 1
    frq = np.fft.fftfreq(signal_length, d=(1 / sampling_rate))  # divide the bins into frequency categories
    frq = frq[range(fft_split)]  # Get single side of the frequency range
    # Do FFT
    Y = np.fft.fft(f(RR_x_new)) / signal_length  # Calculate FFT
    Y = Y[range(min(fft_split, len(Y)))]  # Return one side of the FFT
    lf = np.trapz(abs(Y[(frq >= 0.04) & (frq <= 0.15)]))
    hf = np.trapz(abs(Y[(frq >= 0.16) & (frq <= 0.5)]))
    return lf, hf


def get_bpm(rr_intervals):
    return 60 / np.mean(rr_intervals)


def feature_extraction(sample, sampling_rate=300):
    # Normalize raw signal
    signal = ecg.st.normalize(sample)['signal']

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    order = int(0.3 * sampling_rate)
    filtered, _, _ = ecg.st.filter_signal(signal=signal,
                                          ftype='FIR',
                                          band='bandpass',
                                          order=order,
                                          frequency=[5, 15],
                                          sampling_rate=sampling_rate)

    # segment
    rpeaks, = ecg.hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)

    # correct R-peak locations
    rpeaks, = ecg.correct_rpeaks(signal=filtered,
                                 rpeaks=rpeaks,
                                 sampling_rate=sampling_rate,
                                 tol=0.05)

    templates, rpeaks = ecg.extract_heartbeats(signal=filtered,
                                               rpeaks=rpeaks,
                                               sampling_rate=sampling_rate,
                                               before=0.2,
                                               after=0.4)

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=False)

    # Extract time domain measures
    rpeaks_time = ts[rpeaks]

    rr_intervals = extract_rr_intervals(rpeaks_time)
    rr_diffs = extract_rr_diffs(rr_intervals)
    bpm = get_bpm(rr_intervals)
    ibi = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    sdsd = np.std(rr_diffs)
    rmssd = np.sqrt(np.mean(rr_diffs ** 2))
    nn20 = np.sum([rr_diffs > 0.02]) / len(rr_diffs)
    nn50 = np.sum([rr_diffs > 0.05]) / len(rr_diffs)

    # QRS
    qpeaks = find_q_point(filtered, rpeaks)
    speaks = find_s_point(filtered, rpeaks)

    # Extract wavelet power
    power_spectrum = ecg.st.power_spectrum(filtered, 300)
    frequency = [power_spectrum['freqs'][0], power_spectrum['freqs'][-1]]
    wavelet_energie = ecg.st.band_power(power_spectrum['freqs'], power_spectrum['power'], frequency)['avg_power']

    # Amplitudes
    qamplitudes = filtered[qpeaks]
    ramplitudes = filtered[rpeaks]
    samplitudes = filtered[speaks]

    # QRS duration
    qrs_duration = ts[speaks] - ts[qpeaks]
    qrs_duration_diffs = extract_rr_diffs(qrs_duration)
    iqrs = np.mean(qrs_duration)
    sdqrs = np.std(qrs_duration)
    sdqrsdiffs = np.std(qrs_duration_diffs)
    rmssqrsdiffs = np.sqrt(np.mean(qrs_duration_diffs ** 2))

    extracted_features = [
        bpm, ibi, sdnn, sdsd, rmssd, nn20, nn50, wavelet_energie,
        iqrs, sdqrs, sdqrsdiffs, rmssqrsdiffs,
        np.median(qamplitudes), np.min(qamplitudes), np.max(qamplitudes),
        np.median(ramplitudes), np.min(ramplitudes), np.max(ramplitudes),
        np.median(samplitudes), np.min(samplitudes), np.max(samplitudes)
    ]
    return extracted_features
