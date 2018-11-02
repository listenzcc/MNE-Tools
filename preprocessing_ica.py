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
raw_raw.annotations = mne.Annotations([0], [2], 'BAD')
raw_env = mne.io.RawArray(
    smooth(get_envlop(raw.get_data(), picks), picks),
    raw.info)

raw = raw_raw
events = mne.find_events(raw)

ica = ICA(n_components=0.95, method='fastica')

picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                       stim=False, exclude='bads')

ica.fit(raw, picks=picks, reject=reject)

sources = ica.get_sources(raw)
epochs = mne.Epochs(sources, events=events, event_id=event_ids,
                    tmin=tmin, tmax=tmax,
                    baseline=(tmin, t0))
# plot_evoked(epochs)
d = epochs.get_data()
d_mean = np.mean(d, axis=0)
select = [12, 18, 20, 23, 25, 27]
select = list(range(d.shape[1]))
plt.plot(epochs.times, d_mean[select].transpose())
plt.legend(tuple(str(e) for e in select))
plt.title(str(epochs.event_id))

ica.plot_components(picks=select, show=False)

plt.show()
