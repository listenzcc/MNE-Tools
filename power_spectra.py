# coding: utf-8

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet

from mnetools_zcc import (prepare_raw,
                          get_envlop,
                          get_epochs)

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

train = False
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


def plot_raw(raw, event_ids=event_ids, picks=picks,
             tmin=tmin, tmax=tmax, t0=t0,
             freqs=freqs, n_cycles=n_cycles):
    # get_epochs
    epochs = get_epochs(raw, event_ids, picks,
                        tmin=tmin, tmax=tmax,
                        baseline=(tmin, t0), decim=1)
    # calculate power spectra
    power = tfr_morlet(epochs,
                       freqs=freqs, n_cycles=n_cycles,
                       use_fft=False, decim=1, n_jobs=6)
    # plot epochs
    epochs.average().plot(time_unit='s', show=False)
    # plot power spectra
    power[0].plot_joint(baseline=(tmin, t0),
                        show=False, mode='mean',
                        tmin=tmin, tmax=tmax,
                        timefreqs=timefreqs)
    # return
    return epochs, power


epochs, power = plot_raw(raw_raw)
plt.show()
