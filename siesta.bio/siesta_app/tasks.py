# Standard Library Imports
import datetime
from math import floor, ceil
import os
import uuid
import joblib

# Third-Party Imports
import numpy as np
import pandas as pd
import scipy
import scipy.signal
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d
from twistpy.utils import stransform
from mne.io import read_raw_edf
from sklearn import preprocessing
from scipy.signal import butter, filtfilt, resample
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler

# Django Imports
from django.conf import settings

# Custom Imports
from SIESTA.celery import app as celery_app
from siesta_app.utilities import fitModel,  getPredictions

MEDIA = f'{settings.MEDIA_ROOT}'
STATIC = f'{settings.STATIC_ROOT}'

__all__ = ('celery_app',)

#Feature extraction functions
# Calculates power spectral density
def calculate_psd_and_f(signal, fs, epoch, nfft=256, freq_min=1, freq_max=50, num_freq_bins=100):
    epoch_samples = int(epoch * fs)
    nfft = nfft if nfft >= epoch_samples else epoch_samples

    corr_signal = signal[:len(signal) - (len(signal) % epoch_samples)]
    new_signal = np.reshape(corr_signal, (len(corr_signal) // epoch_samples, epoch_samples))

    fr, p = scipy.signal.welch(new_signal, fs=fs, nperseg=epoch_samples, nfft=nfft, scaling='spectrum')

    common_freqs = np.linspace(freq_min, freq_max, num_freq_bins)
    p_interp = np.zeros((p.shape[0], len(common_freqs)))
    for i in range(p.shape[0]):
        p_interp[i, :] = np.interp(common_freqs, fr, p[i, :])

    return common_freqs, p_interp

# Performs butter bandpass filter
@celery_app.task
def butter_bandpass_features(data, lowcut, highcut, fs, epoch, totalenergy = None, totalamp = None, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')

    filtered_data = filtfilt(b, a, data, axis=1)

    energy = np.sum(filtered_data**2, axis=1)/np.sum(data**2, axis=1)
    amp = np.mean(np.abs(filtered_data), axis=1)/np.mean(np.abs(data), axis=1)

    if totalenergy is not None:
        energy = energy/totalenergy
        amp = amp/totalamp

    return energy, amp

# Creates matrix of extracted features
@celery_app.task
def feature_calculations(signal,signal_label,epoch,fs):
    targetfs = 120
    signal = scipy.signal.resample_poly(signal, targetfs, fs)

    freq, p = calculate_psd_and_f(signal, targetfs, epoch)
    epochfs = epoch*targetfs

    num_epochs = p.shape[0]
    signal = signal[:num_epochs * epochfs]
    s = np.reshape(signal, (num_epochs, epochfs))

    powersum = p

    # Calculate power ranges and standardized power
    powermax = np.max(powersum, axis=1)
    powermin = np.min(powersum, axis=1)
    powerrange = powermax - powermin
    powerstd = ((powersum.T - powermin) / powerrange).T

    # Calculate relative power in different frequency bands:
    delta1 = np.sum(powerstd[:, (freq >= 1) & (freq < 2.5)], axis=1)
    delta2 = np.sum(powerstd[:, (freq >= 2.5) & (freq < 4)], axis=1)
    theta1 = np.sum(powerstd[:, (freq >= 4) & (freq < 6)], axis=1)
    theta2 = np.sum(powerstd[:, (freq >= 6) & (freq < 8)], axis=1)
    alpha1 = np.sum(powerstd[:, (freq >= 8) & (freq < 10)], axis=1)
    alpha2 = np.sum(powerstd[:, (freq >= 10) & (freq < 12)], axis=1)
    sigma1 = np.sum(powerstd[:, (freq >= 12) & (freq < 14)], axis=1)
    sigma2 = np.sum(powerstd[:, (freq >= 14) & (freq < 16)], axis=1)
    beta = np.sum(powerstd[:, (freq >= 16) & (freq < 30)], axis=1)
    gamma = np.sum(powerstd[:, (freq >= 30) & (freq <= 50)], axis=1)

    # Calculate relative ratios for different bands
    deltaratio = delta2/delta1
    alphatheta = (alpha1 + alpha2) / (theta1 + theta2)
    thetadelta = (theta1 + theta2) / (delta1 + delta2)
    alphadelta = (alpha1 + alpha2) / (delta1 + delta2)
    sigmadelta = (sigma1 + sigma2) / (delta1 + delta2)
    betadelta = beta / (delta1 + delta2)
    gammadelta = gamma / (delta1 + delta2)
    lowfreq = np.sum(powerstd[:, (freq >= 1) & (freq <= 20)], axis=1)

    # Sigma Index and Relative Spindle Power (O'Reilly & Nielsen, 2015)
    common_freqs = np.linspace(0, 40, num=41)

    SI = []
    RSP = []

    for i in range(len(s)):
        curr_s = resample(s[i], int(epoch * 80))
        stran_raw = np.abs(stransform(curr_s, 5)[0]).transpose()[50:-50, :]

        freqs_orig = np.linspace(0, 40, stran_raw.shape[1])
        f_interp = interp1d(freqs_orig, stran_raw, kind='linear', axis=1, fill_value="extrapolate")
        stran = f_interp(common_freqs)

        band_11_16 = (common_freqs >= 11) & (common_freqs <= 16)
        band_20_40 = (common_freqs >= 20) & (common_freqs <= 40)
        band_4_10  = (common_freqs >= 4)  & (common_freqs <= 10)
        band_7_5_10= (common_freqs >= 7.5) & (common_freqs <= 10)
        band_1_40  = (common_freqs >= 1)  & (common_freqs <= 40)

        mean_band_20_40 = np.mean(stran[:, band_20_40], axis=1)
        mean_band_4_10  = np.mean(stran[:, band_4_10], axis=1)
        mean_band_7_5_10= np.mean(stran[:, band_7_5_10], axis=1)
        max_band_11_16  = np.max(stran[:, band_11_16], axis=1)

        sum_means = mean_band_20_40 + mean_band_4_10
        SI_val = np.mean(np.where(max_band_11_16 < mean_band_7_5_10, 0, 2 * max_band_11_16 / (sum_means)))
        SI.append(SI_val)

        bin_width = common_freqs[1] - common_freqs[0]
        rsp_val = (np.sum(stran[:, band_11_16], axis=1) * bin_width) / (np.sum(stran[:, band_1_40], axis=1) * bin_width)
        RSP.append(np.mean(rsp_val))

    #Energy and amplitude in different frequency bands
    totalenergy, totalamp = butter_bandpass_features(s, 1, 50, targetfs, epoch)
    delta1energy, delta1amp = butter_bandpass_features(s, 1, 2.5, targetfs, epoch, totalenergy, totalamp)
    delta2energy, delta2amp = butter_bandpass_features(s, 2.5, 4, targetfs, epoch, totalenergy, totalamp)
    theta1energy, theta1amp = butter_bandpass_features(s, 4, 6, targetfs, epoch, totalenergy, totalamp)
    theta2energy, theta2amp = butter_bandpass_features(s, 6, 8, targetfs, epoch, totalenergy, totalamp)
    alpha1energy, alpha1amp = butter_bandpass_features(s, 8, 10, targetfs, epoch, totalenergy, totalamp)
    alpha2energy, alpha2amp = butter_bandpass_features(s, 10, 12, targetfs, epoch, totalenergy, totalamp)
    sigma1energy, sigma1amp = butter_bandpass_features(s, 12, 14, targetfs, epoch, totalenergy, totalamp)
    sigma2energy, sigma2amp = butter_bandpass_features(s, 14, 16, targetfs, epoch, totalenergy, totalamp)
    betaenergy, betaamp = butter_bandpass_features(s, 16, 30, targetfs, epoch, totalenergy, totalamp)
    gammaenergy, gammaamp = butter_bandpass_features(s, 30, 50, targetfs, epoch, totalenergy, totalamp)

    # Calculate spectral edge (90%) and mean (50%)
    s_cumsum = np.cumsum(powersum, axis=1)

    spectral90 = 0.9 * np.sum(powersum, axis=1)
    l_edge = np.argmax(s_cumsum >= spectral90[:, None], axis=1)
    spectraledge = np.take(freq, l_edge)

    spectral50 = 0.5 * np.sum(powersum, axis=1)
    l_mean = np.argmax(s_cumsum >= spectral50[:, None], axis=1)
    spectralmean50 = np.take(freq, l_mean)

    ## Calculate the spectral mean and the spectral entropy (essentially the spectral power distribution):
    spectralmean = np.mean(powerstd, axis=1)
    spectralentropy = -(np.sum((powerstd+1e-8) * np.log(powerstd+1e-8), axis=1)) / np.log(powerstd.shape[1])

    ## Calculate skewness and kurtosis
    skewness = skew(s, axis=1)
    kurt = kurtosis(s, axis=1)

    ## Calculate zero cross
    zerocross = (np.diff(np.sign(preprocessing.maxabs_scale(s, axis=1))) != 0).sum(axis=1) / epochfs

    # Calculate relative time-domain features:
    maxs = np.amax(s, axis=1)
    mins = np.amin(s, axis=1)
    rms = np.sqrt(np.mean(s**2, axis=1))
    peaktopeak = (np.amax(s, axis=1) - np.amin(s, axis=1))
    arv = np.mean(np.abs(s), axis=1)
    amplitude = np.mean(np.abs(s), axis=1)
    amplitudemed = np.median(np.abs(s), axis=1)
    signalvar = np.var(np.abs(s), axis=1, ddof=1)

    #Time-domain features, but normalized
    norm_maxs = maxs / rms
    norm_mins = mins / rms
    norm_rms = rms / arv
    norm_peaktopeak = peaktopeak / rms
    norm_arv = arv / rms
    norm_amplitudemed = amplitudemed / rms
    cv = np.sqrt(signalvar) / rms

    feature_list = np.column_stack([
        delta1, delta2, theta1, theta2, alpha1, alpha2, sigma1, sigma2, beta, gamma,
        deltaratio, alphatheta, thetadelta, alphadelta, sigmadelta, betadelta, gammadelta,
        lowfreq, SI, RSP,
        delta1energy, delta2energy, theta1energy, theta2energy,
        alpha1energy, alpha2energy, sigma1energy, sigma2energy,
        betaenergy, gammaenergy,
        delta1amp, delta2amp, theta1amp, theta2amp,
        alpha1amp, alpha2amp, sigma1amp, sigma2amp,
        betaamp, gammaamp,
        spectraledge, spectralmean50, spectralmean, spectralentropy,
        skewness, kurt, zerocross,
        #maxs, mins, rms, peaktopeak, arv, amplitude, amplitudemed, signalvar,
        norm_maxs, norm_mins, norm_rms, norm_peaktopeak, norm_arv, norm_amplitudemed, cv
    ])

    feature_labels = [
        'delta1', 'delta2', 'theta1', 'theta2', 'alpha1', 'alpha2', 'sigma1', 'sigma2', 'beta', 'gamma',
        'deltaratio', 'alphatheta', 'thetadelta','alphadelta', 'sigmadelta', 'betadelta', 'gammadelta',
        'lowfreq', 'SI', 'RSP',
        'delta1energy', 'delta2energy', 'theta1energy', 'theta2energy',
        'alpha1energy', 'alpha2energy', 'sigma1energy', 'sigma2energy',
        'betaenergy', 'gammaenergy',
        'delta1amp', 'delta2amp', 'theta1amp', 'theta2amp',
        'alpha1amp', 'alpha2amp', 'sigma1amp', 'sigma2amp',
        'betaamp', 'gammaamp',
        'spectraledge', 'spectralmean50', 'spectralmean', 'spectralentropy',
        'skewness', 'kurt', 'zerocross',
        #'maxs', 'mins', 'rms', 'peaktopeak', 'arv', 'amplitude', 'amplitudemed', 'signalvar',
        'maxs_norm', 'mins_norm', 'rms_norm', 'peaktopeak_norm', 'arv_norm', 'amplitudemed_norm', 'cv'

    ]

    feature_labels = [f"{signal_label}_{label}" for label in feature_labels]

    return feature_list, feature_labels

@celery_app.task
# Opens .edf file containing signal data and extracts ecog and emg data
def CreateFeaturesDataFrame(self, session_key, filename, fs, epoch, ECoG1_chan, EMG_chan, ECoG2_chan):
    filepath = os.path.join(MEDIA, session_key, filename)
    f = read_raw_edf(filepath, preload=False)

    current_channels = f.info['ch_names']
    new_channel_names = {}

    new_channel_names[current_channels[ECoG1_chan]] = 'ECoG1'
    if ECoG2_chan: new_channel_names[current_channels[ECoG2_chan]] = 'ECoG2'
    new_channel_names[current_channels[EMG_chan]] = 'EMG'

    f.rename_channels(new_channel_names)
    f.pick(picks=['ECoG1', 'ECoG2', 'EMG'] if ECoG2_chan else ['ECoG1', 'EMG'])
    f.load_data()

    signal = f.get_data('ECoG1').flatten()
    signal = signal[:(len(signal) - (len(signal) % (epoch * fs)))]
    s = np.reshape(signal, (int(len(signal)/(epoch*fs)), epoch * fs))
    useRows = (np.amax(s, axis=1) - np.amin(s, axis=1)) != 0

    f = f.filter(1, 70, picks = 'ECoG1')
    if ECoG2_chan: f = f.filter(1, 70, picks = 'ECoG2')
    f = f.filter(3, 100, picks = 'EMG')

    start_date = f.info['meas_date']
    fd = len(f)/fs
    file_end = start_date + datetime.timedelta(seconds = fd)

    num_epochs = floor(fd/epoch)
    num_epochs_each_iteration = 500

    #"Total epochs =", floor(num_epochs/num_epochs_each_iteration))

    feature_matrix = []

    for curr_epoch in range(0, ceil(num_epochs / num_epochs_each_iteration)):
        offset = curr_epoch * num_epochs_each_iteration

        if (num_epochs - offset) < num_epochs_each_iteration:
            num_epochs_each_iteration = num_epochs - offset

        offset_epoch = offset * fs * epoch

        ecog1_data = f.get_data('ECoG1', start=offset_epoch, stop=offset_epoch + fs * epoch * num_epochs_each_iteration).flatten()
        emg_data = f.get_data('EMG', start=offset_epoch, stop=offset_epoch + fs * epoch * num_epochs_each_iteration).flatten()
        if ECoG2_chan: ecog2_data = f.get_data('ECoG2', start=offset_epoch, stop=offset_epoch + fs * epoch * num_epochs_each_iteration).flatten()

        ecog1_features = feature_calculations(ecog1_data, 'ECoG_F', epoch, fs)
        emg_features = feature_calculations(emg_data, 'EMG', epoch, fs)
        if ECoG2_chan: ecog2_features = feature_calculations(ecog2_data, 'ECoG_P', epoch, fs)

        feature_matrix.append({
            "epoch_range": list(range(offset, num_epochs_each_iteration + offset)),
            "ecog1": ecog1_features[0],
            "ecog2": ecog2_features[0] if ECoG2_chan else None,
            "emg": emg_features[0]
        })

        curr_state = "{}".format(floor(100*offset/num_epochs))
        self.update_state(state=curr_state, meta = {'status':'progressing'})

    del(f)

    ecog1_labels = ecog1_features[1]
    ecog2_labels = ecog2_features[1] if ECoG2_chan else None
    emg_labels = emg_features[1]

    all_ecog1 = np.concatenate([f["ecog1"] for f in feature_matrix])
    all_emg = np.concatenate([f["emg"] for f in feature_matrix])

    if ECoG2_chan:
        all_ecog2 = np.concatenate([f["ecog2"] for f in feature_matrix])
        features = np.column_stack((all_ecog1, all_ecog2, all_emg))
        labels = np.hstack([ecog1_labels, ecog2_labels, emg_labels])
    else:
        features = np.column_stack((all_ecog1, all_emg))
        labels = np.hstack([ecog1_labels, emg_labels])

    current_features = features[3:, :]
    previous_features = [
        features[3 - i: features.shape[0] - i, :]
        for i in range(1, 4)
    ]
    final_features = np.hstack([current_features] + previous_features)

    final_labels = labels
    for i in range(1, 4):
        for item in labels:
            new_labels = f"prev_{i}_{item}"
            final_labels = np.hstack([final_labels, new_labels])

    # iterates by time step until end of file
    step = datetime.timedelta(seconds=epoch)
    time_stamps = []
    while start_date < file_end:
        time_stamps.append(start_date.strftime('%Y-%m-%d %H:%M:%S'))
        start_date += step
    time_stamps = time_stamps[3:]
    useRows = useRows[3:]

    data = pd.DataFrame(final_features,columns=final_labels)

    if (len(data) != len(time_stamps)):
        time_stamps = time_stamps[:len(data)]
        useRows = useRows[:len(data)]

    data['time_stamps'] = time_stamps
    data['useRows'] = useRows

    os.remove(filepath)

    return data


@celery_app.task(bind=True)
def download_feat(self, session_key, filename, fs, epoch, ECoG1_chan, EMG_chan, ECoG2_chan=None):
    data_frame = CreateFeaturesDataFrame(self, session_key, filename, fs, epoch, ECoG1_chan, EMG_chan, ECoG2_chan)
    return data_frame.to_json()


@celery_app.task(bind=True)
def train_new_model(self, folder, files, include_training_data, channels):
    self.update_state(state='FITTING', meta={'status': 'fitting'})

    chunk_size = 10000
    model_wake = SGDClassifier(penalty='l2', loss='log_loss', max_iter=10000, warm_start=True)
    model_sleep = SGDClassifier(penalty='l2', loss='log_loss', max_iter=10000, warm_start=True)

    def load_and_process_data_in_chunks(channels, files, chunk_size=10000):
        if include_training_data:
            try:
                channels = int(channels)
                if channels == 1:
                    chunk = pd.read_csv(os.path.join(STATIC, "admin/model_data/training_frontal.csv"), chunksize=chunk_size)
                elif channels == 2:
                    chunk = pd.read_csv(os.path.join(STATIC, "admin/model_data/training_both.csv"), chunksize=chunk_size)
                else:
                    raise ValueError("Invalid number of channels. Only 1 or 2 are valid options.")
            except ValueError:
                raise ValueError("Invalid input for 'channels'.")

            for data_chunk in chunk:
                yield data_chunk

        if len(files) > 0:
            for f in files:
                file_path = os.path.join(MEDIA, folder, f)
                chunk = pd.read_csv(file_path, chunksize=chunk_size)
                for data_chunk in chunk:
                    useRows = data_chunk.iloc[:, -2].astype(bool)
                    filtered_chunk = data_chunk[useRows].copy()
                    filtered_chunk.drop(filtered_chunk.columns[-2], axis=1, inplace=True)

                    yield filtered_chunk

    scaler = None
    for data_chunk in load_and_process_data_in_chunks(1, files):
        data_values = data_chunk.values

        if scaler is None:
            useRows = data_values[:, -2].astype(bool)
            scaler = MinMaxScaler().fit(data_values[useRows, :-2])
        model_wake, model_sleep = fitModel(data_values, model_wake, model_sleep, scaler, chunk_size)

    model_name = str(uuid.uuid4()) + '.file'
    joblib.dump({'model': {'Step 1': model_wake, 'Step 2': model_sleep}, 'scaler': scaler}, os.path.join(MEDIA, folder, model_name), compress=3)

    return model_name


@celery_app.task(bind=True)
def sleep_prediction(self, folder, fileName, modelType, modelName):
    try:
        data = pd.read_csv(os.path.join(MEDIA, folder, fileName), parse_dates=[-2])
        if data is None:
            raise ValueError("File not found.")

        if modelType == 'new':
            modelPath = os.path.join(MEDIA, folder, modelName)
        elif modelType == 'pre-trained':
            if len(data.columns) == 434:
                modelPath = os.path.join(STATIC, 'admin', 'model_data', 'model_frontal.file')
            elif len(data.columns) == 650:
                modelPath = os.path.join(STATIC, 'admin', 'model_data', 'model_both.file')
            else:
                raise ValueError("File size incorrect.")
        else:
            raise ValueError("Unknown Error.")

        if not os.path.exists(modelPath):
            raise ValueError(f"Model not found at path: {modelPath}")

        with open(modelPath, 'rb') as f:
            model = joblib.load(f)

        scored_data = getPredictions(data, model)
        if scored_data is None:
            raise ValueError("Error in prediction.")

        return scored_data.to_json(orient='records')

    except FileNotFoundError as e:
        print(f"File error: {e}")
    except pd.errors.ParserError as e:
        print(f"CSV Parsing error: {e}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")