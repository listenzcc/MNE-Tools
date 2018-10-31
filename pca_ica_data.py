# coding: utf-8

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet

from mnetools_zcc import (prepare_raw,
                          get_envlop,
                          get_epochs)

from sklearn.decomposition import PCA, FastICA, SparsePCA
from sklearn.preprocessing import normalize
from mne.decoding import UnsupervisedSpatialFilter

smooth_kernel = 1/200+np.array(range(200))*0


def smooth(x, picks, y=smooth_kernel):
    return x
    for j in picks:
        x[j] = np.convolve(x[j], y, 'same')
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
    tmin, t0, tmax = -0.25, 0, 1.25
    timefreqs = [(0.8, 9.5), (1.0, 9.5), (0.2, 1.3)]
else:
    fname = fname_testing_list[3]
    event_ids = dict(ort45a=8, ort135a=16,
                     ort45b=32, ort135b=64)
    tmin, t0, tmax = -0.4, -0.2, 1.25
    timefreqs = [(-0.3, 1.1), (0.5, 1.1), (0.7, 1.1)]

freq_l, freq_h = 1, 15
freqs = np.logspace(*np.log10([1, 12]), num=20)
n_cycles = freqs / 2.
event_list = list(e for e in event_ids.values())

raw, picks = prepare_raw(fname)
raw.filter(freq_l, freq_h, fir_design='firwin')
raw_raw = mne.io.RawArray(
    smooth(raw.get_data(), picks),
    raw.info)
raw_env = mne.io.RawArray(
    smooth(get_envlop(raw.get_data(), picks), picks),
    raw.info)

average = False
pca = UnsupervisedSpatialFilter(PCA(30), average=average)
ica = UnsupervisedSpatialFilter(FastICA(30), average=average)
spca = UnsupervisedSpatialFilter(SparsePCA(30, alpha=0.01), average=average)

# get_epochs
epochs = get_epochs(raw, event_ids, picks,
                    tmin=tmin, tmax=tmax,
                    baseline=(tmin, t0), decim=1)

data = epochs.get_data()


def normalizedata(data=data):
    shape = data.shape
    dd = data.reshape(-1, shape[1])
    return normalize(dd.transpose()).transpose().reshape(shape)


data = normalizedata()

print('PCA')
data_pca = pca.fit_transform(data)
print('ICA')
data_ica = ica.fit_transform(data)
print('sPCA')
data_spca = spca.fit_transform(data)


def plot(row, col, data, epochs=epochs, event_list=event_list):
    fig, axes = plt.subplots(row, col)
    for j in range(len(event_list)):
        ort_idx = event_list[j]
        idx_h, idx_l = int(j/col), (j % col)
        x = data[epochs.events[:, 2] == ort_idx]
        axes[idx_h][idx_l].plot(
            epochs.times, np.mean(x, axis=0).transpose())
        axes[idx_h][idx_l].set_title(ort_idx)


print('Plotting')
plot(3, 2, data_pca)
plot(3, 2, data_ica)
plot(3, 2, data_spca)
'''
fig, axes = plt.subplots(3, 2)
for j in range(len(event_list)):
    ort_idx = event_list[j]
    idx_h, idx_l = int(j/2), (j % 2)
    x = data_pca[epochs.events[:, 2] == ort_idx]
    axes[idx_h][idx_l].plot(
        epochs.times, np.mean(x, axis=0).transpose())
    axes[idx_h][idx_l].set_title(ort_idx)

fig, axes = plt.subplots(3, 2)
for j in range(len(event_list)):
    ort_idx = event_list[j]
    idx_h, idx_l = int(j/2), (j % 2)
    x = data_ica[epochs.events[:, 2] == ort_idx]
    axes[idx_h][idx_l].plot(
        epochs.times, np.mean(x, axis=0).transpose())
    axes[idx_h][idx_l].set_title(ort_idx)
'''

plt.show()
