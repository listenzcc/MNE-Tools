# coding: utf-8

import mne
import numpy as np
import matplotlib.pyplot as plt
from mnetools_zcc import prepare_raw, get_envlop, get_epochs

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA, FastICA

from mne.decoding import GeneralizingEstimator
from mne.decoding import UnsupervisedSpatialFilter

import itertools

smooth_kernel = 1/20+np.array(range(20))*0


def smooth(x, y=smooth_kernel):
    for j in range(len(x)):
        for k in range(len(x[j])):
            x[j][k] = np.convolve(x[j][k], y, 'same')
    return x


def stack_3Ddata(a, b):
    # Stack 3D data on first dimension
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

event_ids = dict(ort1=2, ort2=33)

freq_l, freq_h = 0.03, 33  # 1, 15

for pare in itertools.combinations([2, 6, 9, 14, 17, 33], r=2):
    event_ids = dict()
    event_ids['ort1'] = pare[0]
    event_ids['ort2'] = pare[1]

    data_raw = []
    label_raw = []
    for j in range(5):
        fname = fname_list[j]
        print(fname)
        raw, picks = prepare_raw(fname)
        raw.filter(freq_l, freq_h, fir_design='firwin')
        # raw_custom = mne.io.RawArray(get_envlop(raw.get_data(), picks), raw.info)
        raw_custom = mne.io.RawArray(raw.get_data(), raw.info)
        epochs = get_epochs(raw_custom, event_ids, picks, decim=1)
        data_raw.append(smooth(epochs.get_data()))
        label_raw.append(epochs.events[:, 2])

    # Init pca and ica object
    pca = UnsupervisedSpatialFilter(PCA(30), average=False)
    ica = UnsupervisedSpatialFilter(FastICA(30), average=False)

    scores_cv = []
    for test_idx in range(5):
        print('crossing validation in %d' % test_idx)
        # Seperate datasets into train and test, test
        data_all = data_raw.copy()
        label_all = label_raw.copy()

        data_test = data_all.pop(test_idx)
        label_test = label_all.pop(test_idx)

        # Seperate datasets into train and test, train
        data_train = data_all.pop()
        label_train = label_all.pop()

        while data_all.__len__():
            data_train = stack_3Ddata(data_train, data_all.pop())
            label_train = np.hstack((label_train, label_all.pop()))

        # Fit pca model for train dataset (fitting) and test dataset (transforming)
        train_data = pca.fit_transform(data_train)
        test_data = pca.transform(data_test)

        # We will train and test using leave one run out protocal
        # and test on all directions from the leaved run.
        # clf = make_pipeline(StandardScaler(), LogisticRegression())
        clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
        # Scoring='roc_auc' can only be used in two-classes classification
        n_jobs = 5  # para on 5
        time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=n_jobs)

        # Fit classifiers on the epochs.
        # Note that the experimental condition y indicates directions
        time_gen.fit(X=data_train, y=label_train)

        # Score on the epochs
        scores_ = time_gen.score(X=data_test, y=label_test)
        scores_cv.append(scores_)

    # Mean scores
    scores = sum(e for e in scores_cv) / 5
    # Plotting layout
    fig = plt.figure(figsize=(15, 6))

    # Plot Generalization over time
    ax = fig.add_subplot(122)
    im = ax.matshow(scores, vmin=0, vmax=1., cmap='RdBu_r', origin='lower',
                    extent=epochs.times[[0, -1, 0, -1]])
    ax.axhline(0., color='k')
    ax.axvline(0., color='k')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Generalization across time and condition')
    plt.colorbar(im, ax=ax)

    # Plot Decoding over time
    ax = fig.add_subplot(121)
    im = ax.plot(epochs.times, np.diag(scores), label='score')
    ax.axhline(.5, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC')
    ax.legend()
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title('Diag'+', '.join(idx2angle[e] for e in event_ids.values()))

    # Plotting
    plt.savefig('pics/cv5_'+'_'.join(idx2angle[e]
                                     for e in event_ids.values())+'.png')
    plt.close()
