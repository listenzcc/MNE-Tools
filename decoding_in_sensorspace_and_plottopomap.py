# coding: utf-8

import numpy as np

import os
import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mnetools_zcc import (prepare_raw, get_envlop, get_epochs,
                          plot_evoked, reject)

smooth_kernel = 1/200+np.array(range(200))*0


def smooth(x, picks, y=smooth_kernel):
    for j in picks:
        None
        # x[j] = (x[j] - np.mean(x[j])) / np.std(x[j])
        # x[j] = np.convolve(x[j], y, 'same')
    return x


# QYJ, ZYF
filedir = 'D:/BeidaShuju/rawdata/QYJ'
fname_training_list = list(os.path.join(
    filedir, 'MultiTraining_%d_raw_tsss.fif' % j)
    for j in range(1, 6))
fname_testing_list = list(os.path.join(
    filedir, 'MultiTest_%d_raw_tsss.fif' % j)
    for j in range(1, 9))


train = True
if train:
    fname = fname_training_list[3]
    event_ids = dict(ort015=2,  ort045=6,  ort075=9,
                     ort105=14, ort135=17, ort165=33)
    event_ids = dict(ort135=17, ort045=6)
    tmin, t0, tmax = -0.25, 0, 1.25
    timefreqs = [(0.8, 9.5), (1.0, 9.5), (0.2, 1.3)]
else:
    fname = fname_testing_list[3]
    event_ids = dict(ort45a=8, ort135a=16,
                     ort45b=32, ort135b=64)
    tmin, t0, tmax = -0.4, -0.2, 1.25
    timefreqs = [(-0.3, 1.1), (0.5, 1.1), (0.7, 1.1)]

freq_l, freq_h = 1, 15
event_list = list(e for e in event_ids.values())

raw, picks = prepare_raw(fname)
raw.filter(freq_l, freq_h, fir_design='firwin')
raw_raw = mne.io.RawArray(
    smooth(raw.get_data(), picks),
    raw.info)
raw_env = mne.io.RawArray(
    smooth(get_envlop(raw.get_data(), picks), picks),
    raw.info)

raw = raw_raw
epochs = get_epochs(raw,
                    event_id=event_ids, picks=picks,
                    tmin=tmin, tmax=tmax, baseline=(tmin, t0))
labels = epochs.events[:, -1]
evoked = epochs.average()

from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from mne.decoding import CSP

n_components = 3  # pick some components
csp = CSP(n_components=n_components, norm_trace=False)
svc = SVC(C=1, kernel='linear')

# Define a monte-carlo cross-validation generator (reduce variance):
cv = ShuffleSplit(n_splits=10, test_size=0.2)
scores = []
epochs_data = epochs.get_data()

for train_idx, test_idx in cv.split(labels):
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data[train_idx], y_train)
    X_test = csp.transform(epochs_data[test_idx])

    # fit classifier
    svc.fit(X_train, y_train)

    scores.append(svc.score(X_test, y_test))

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))

# Or use much more convenient scikit-learn cross_val_score function using
# a Pipeline
from sklearn.pipeline import Pipeline  # noqa
from sklearn.model_selection import cross_val_score  # noqa
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
clf = Pipeline([('CSP', csp), ('SVC', svc)])
scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)
print(scores.mean())  # should match results above

# And using reuglarized csp with Ledoit-Wolf estimator
csp = CSP(n_components=n_components, reg='ledoit_wolf', norm_trace=False)
clf = Pipeline([('CSP', csp), ('SVC', svc)])
scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)
print(scores.mean())  # should get better results than above

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)
data = csp.patterns_
fig, axes = plt.subplots(1, 4)
for idx in range(4):
    mne.viz.plot_topomap(data[idx], evoked.info, axes=axes[idx], show=False)
fig.suptitle('CSP patterns')
fig.tight_layout()
mne.viz.utils.plt_show()
