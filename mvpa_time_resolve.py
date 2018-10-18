# coding: utf-8

import mne
import numpy as np
import matplotlib.pyplot as plt
from mnetools_zcc import prepare_raw, get_envlop, get_epochs

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_classif

from mne.decoding import GeneralizingEstimator
from mne.decoding import UnsupervisedSpatialFilter
from mne.decoding import (cross_val_multiscore, LinearModel, SlidingEstimator,
                          get_coef)

import itertools

pca = UnsupervisedSpatialFilter(PCA(30), average=False)

smooth_kernel = 1/200+np.array(range(200))*0


def smooth(x, picks, y=smooth_kernel):
    for j in picks:
        x[j] = np.convolve(x[j], y, 'same')
    return x


def stack_3Ddata(a, b):
    # Stack 3D data on first dimension
    if len(a) == 0:
        return b
    sa = a.shape
    sb = b.shape
    return np.vstack((a.reshape(sa[0], sa[1]*sa[2]),
                      b.reshape(sb[0], sb[1]*sb[2]))).reshape(sa[0]+sb[0], sa[1], sa[2])


fname_list = [
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTraining_1_raw_tsss.fif',
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTraining_2_raw_tsss.fif',
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTraining_3_raw_tsss.fif',
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTraining_4_raw_tsss.fif',
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTraining_5_raw_tsss.fif',
]

idx2angle = {2: '15', 6: '45', 9: '75',
             14: '105', 17: '135', 33: '165'}

event_ids = dict(ort015=2,  ort045=6,  ort075=9,
                 ort105=14, ort135=17, ort165=33)

event_list = list(e for e in event_ids.values())

freq_l, freq_h = 1, 15

data_all = []
label_all = []
for j in range(len(fname_list)):
    fname = fname_list[j]
    print(fname)
    raw, picks = prepare_raw(fname)
    raw.filter(freq_l, freq_h, fir_design='firwin')
    data_fif = []
    label_fif = []
    for k in range(len(event_list)):
        event_ids = dict()
        event_ids['ort'] = event_list[k]
        print(event_ids.items())
        # raw_custom = mne.io.RawArray(smooth(raw.get_data(), picks), raw.info)
        raw_custom = mne.io.RawArray(
            smooth(get_envlop(raw.get_data(), picks), picks), raw.info)

        epochs = get_epochs(raw_custom, event_ids, picks, decim=10)
        data_fif.append(epochs.get_data())
        label_fif.append(epochs.events[:, 2])
    data_all.append(data_fif)
    label_all.append(label_fif)

n_class = 2
scores_store = []
for pare in itertools.combinations([0, 1, 2, 3, 4, 5], r=n_class):
    print(pare)
    data_all_copy = data_all.copy()
    label_all_copy = label_all.copy()

    X = []
    y = []
    for j in range(len(fname_list)):
        for k in pare:
            print('.', end='')
            X = stack_3Ddata(X, data_all_copy[j][k])
            y = np.hstack((y, label_all_copy[j][k]))
    print('.')

    clf = make_pipeline(StandardScaler(),  # z-score normalization
                        LinearModel(LinearSVC(penalty='l2')))
    # LinearModel(LogisticRegression(penalty='l1')))
    time_decod = SlidingEstimator(clf, scoring='roc_auc')
    scores = cross_val_multiscore(time_decod, X, y, cv=10, n_jobs=6)
    scores_store.append([pare, scores])

    fig, ax = plt.subplots(1)
    ax.plot(epochs.times, scores.mean(0), label='score')
    ax.axhline(1/n_class, color='k', linestyle='--', label='chance')
    ax.axvline(0, color='k')
    pic_name = 'Diag'+', '.join(idx2angle[label_all[0][e][0]] for e in pare)
    ax.set_title(pic_name)
    plt.legend()
    plt.savefig('pics/' + pic_name)

np.save('pics/scores_store', scores_store)
plt.close('all')
