# coding: utf-8

import mne
import numpy as np
import matplotlib.pyplot as plt
from mnetools_zcc import prepare_raw, get_envlop, get_epochs

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, FastICA

from mne.decoding import GeneralizingEstimator
from mne.decoding import UnsupervisedSpatialFilter

fname_list = [
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTraining_1_raw_tsss.fif',
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTraining_2_raw_tsss.fif',
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTraining_3_raw_tsss.fif',
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTraining_4_raw_tsss.fif',
    'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTraining_5_raw_tsss.fif',
]

event_ids = dict(ort015=2,  ort045=6,  ort075=9,
                 ort105=14, ort135=17, ort165=33)
event_ids = dict(ort1=2, ort2=14)

freq_l, freq_h = 1, 15

data_all = []
label_all = []

for j in range(5):
    fname = fname_list[j]
    print(fname)
    raw, picks = prepare_raw(fname)
    raw.filter(freq_l, freq_h, fir_design='firwin')
    raw_custom = mne.io.RawArray(get_envlop(raw.get_data(), picks), raw.info)
    epochs = get_epochs(raw_custom, event_ids, picks)
    data_all.append(epochs.get_data())
    label_all.append(epochs.events[:, 2])

# Seperate datasets into train and test, test
test_idx = 1
data_test = data_all.pop(test_idx)
label_test = label_all.pop(test_idx)
# Seperate datasets into train and test, train
data_train = data_all.pop()
label_train = label_all.pop()


def stack_3Ddata(a, b):
    # Stack 3D data on first dimension
    sa = a.shape
    sb = b.shape
    return np.vstack((a.reshape(sa[0], sa[1]*sa[2]),
                      b.reshape(sb[0], sb[1]*sb[2]))).reshape(sa[0]+sb[0], sa[1], sa[2])


while data_all.__len__():
    data_train = stack_3Ddata(data_train, data_all.pop())
    label_train = np.hstack((label_train, label_all.pop()))

# Init pca and ica object
pca = UnsupervisedSpatialFilter(PCA(30), average=False)
ica = UnsupervisedSpatialFilter(FastICA(30), average=False)

# Fit pca model for train dataset (fitting) and test dataset (transforming)
train_data = pca.fit_transform(data_train)
test_data = pca.transform(data_test)

# We will train and test using leave one run out protocal
# and test on all directions from the leaved run.
clf = make_pipeline(StandardScaler(), LogisticRegression())
# Scoring='roc_auc' can only be used in two-classes classification
n_jobs = 5  # para on 5
time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=n_jobs)

# Fit classifiers on the epochs.
# Note that the experimental condition y indicates directions
time_gen.fit(X=data_train, y=label_train)

# Score on the epochs
scores = time_gen.score(X=data_test, y=label_test)

# Plotting layout
fig, ax_layout = plt.subplots(1, 2)

# Plot Generalization over time
ax = ax_layout[1]
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
ax = ax_layout[0]
im = ax.plot(epochs.times, np.diag(scores), label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Decoding MEG sensors over time')

# Plotting
plt.show()
