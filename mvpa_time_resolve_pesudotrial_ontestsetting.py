# coding: utf-8

import mne
from mne.time_frequency import tfr_morlet
from mne.decoding import GeneralizingEstimator
from mne.decoding import UnsupervisedSpatialFilter
from mne.decoding import (cross_val_multiscore, LinearModel, SlidingEstimator,
                          get_coef)

import numpy as np
import matplotlib.pyplot as plt
from mnetools_zcc import prepare_raw, get_envlop, get_epochs, stack_3Ddata
from pick_good_sensors import good_sensors

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_classif


import itertools

pca = UnsupervisedSpatialFilter(PCA(30), average=False)

smooth_kernel = 1/200+np.array(range(200))*0


def smooth(x, picks, y=smooth_kernel):
    for j in picks:
        x[j] = np.convolve(x[j], y, 'same')
    return x


reject = dict(mag=5e-12, grad=4000e-13)
tmin, tmax = -0.50, 1.25


def epochs_data_2_power(raw, events, picks,
                        tmin=tmin, tmax=tmax, reject=reject):
    freqs = np.logspace(*np.log10([1, 5]), num=20)
    n_cycles = freqs / 2.
    data_out = []
    for e in range(len(events[:, 2])):
        event_ids = dict(x=events[e][2])
        epochs = mne.Epochs(raw, np.reshape(events[e], (1, 3)), event_ids,
                            tmin=tmin, tmax=tmax, picks=picks,
                            baseline=(tmin, 0), reject=reject, preload=True)
        power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                                use_fft=True, return_itc=True,
                                decim=1, n_jobs=6)
        data_power = power.data.transpose((1, 0, 2))
        shape = data_power.shape
        data_out = stack_3Ddata(data_out, np.reshape(
            np.mean(data_power, 0), (1, shape[1], shape[2])))
    return data_out


# QYJ, ZYF
fname_list = [
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTest_1_raw_tsss.fif',
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTest_2_raw_tsss.fif',
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTest_3_raw_tsss.fif',
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTest_4_raw_tsss.fif',
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTest_5_raw_tsss.fif',
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTest_6_raw_tsss.fif',
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTest_7_raw_tsss.fif',
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTest_8_raw_tsss.fif',
]

idx2angle = {8: '8', 16: '16', 32: '32', 64: '64'}

event_ids = dict(ort08=8,  ort16=16,  ort32=32, ort64=64)

event_list = list(e for e in event_ids.values())

freq_l, freq_h = 1, 15

data_all = []
label_all = []
for j in range(len(fname_list)):
    fname = fname_list[j]
    print(fname)
    raw, picks = prepare_raw(fname)
    # sensors, picks = good_sensors(raw.ch_names)
    raw.filter(freq_l, freq_h, fir_design='firwin')
    data_fif = []
    label_fif = []
    for k in range(len(event_list)):
        event_ids = dict()
        event_ids['ort'] = event_list[k]
        print(event_ids.items())
        raw_raw = mne.io.RawArray(
            smooth(raw.get_data(), picks), raw.info)
        raw_env = mne.io.RawArray(
            smooth(get_envlop(raw.get_data(), picks), picks), raw.info)

        epochs = get_epochs(raw_raw, event_ids, picks,
                            tmin=tmin, tmax=tmax,
                            baseline=(tmin, -0.25), decim=1)
        epochs_env = get_epochs(raw_env, event_ids, picks,
                                tmin=tmin, tmax=tmax,
                                baseline=(tmin, -0.25), decim=1)
        data_fif.append(np.hstack(
            (epochs.get_data(), epochs_env.get_data())))
        # data_fif.append(epochs.get_data())
        label_fif.append(epochs.events[:, 2])

        # data_fif.append(epochs_data_2_power(
        #     raw_custom, epochs.events, picks=picks))

    data_all.append(data_fif)
    label_all.append(label_fif)

n_class = 2
scores_store = []
for pare in itertools.combinations([0, 1, 2, 3], r=n_class):
    print(pare)
    data_all_copy = data_all.copy()
    label_all_copy = label_all.copy()

    XX = []
    yy = []
    for j in range(len(fname_list)):
        for k in pare:
            print('.', end='')
            XX = stack_3Ddata(XX, data_all_copy[j][k])
            yy = np.hstack((yy, label_all_copy[j][k]))

    X = []
    y = []
    unique_yy = list(set(yy))
    for u in unique_yy:
        X_tmp = XX[yy == u]
        np.random.shuffle(X_tmp)
        for uu in range(20):
            X = stack_3Ddata(X, np.reshape(
                np.mean(X_tmp[uu*6+0:uu*6+6], 0), (1, 612, 1751)))
            y = np.hstack((y, u))

    # stophere
    print('.')

    clf = make_pipeline(StandardScaler(),  # z-score normalization
                        PCA(n_components=30),
                        # LinearModel(LinearSVC(penalty='l2'))
                        LinearModel(LogisticRegression(penalty='l1'))
                        )
    time_decod = SlidingEstimator(clf, scoring='roc_auc')
    scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=6)
    scores_store.append([pare, scores])

    fig, ax = plt.subplots(1)
    ax.plot(epochs.times, scores.mean(0), label='score')
    ax.axhline(1/n_class, color='k', linestyle='--', label='chance')
    ax.axvline(0, color='k')
    pic_name = 'Diag'+', '.join(idx2angle[label_all[0][e][0]] for e in pare)
    ax.set_title(pic_name)
    plt.legend()
    plt.savefig('pics/' + pic_name)

np.save('pics/epochs_times', epochs.times)
np.save('pics/scores_store', scores_store)
plt.close('all')
