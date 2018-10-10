# coding: utf-8

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack


def get_envlop(x):
    y = fftpack.hilbert(x)
    return np.sqrt(x**2 + y**2)


def get_hilbert(data, picks):
    for j in picks:
        data[j] = get_envlop(data[j])
    return data


# Settings
tmin, tmax = -0.25, 1.25
freq_l, freq_h = 1, 15

# Load raw data
raw_fname = 'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTraining_1_raw_tsss.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw = mne.io.RawArray(raw.get_data(), raw.info)

# Picks
picks = mne.pick_types(raw.info,
                       meg=True, stim=False,
                       eeg=False, eog=False,
                       exclude='bads')

# Filter data
raw.filter(freq_l, freq_h, fir_design='firwin')

# Custom raw
raw_custom = mne.io.RawArray(get_hilbert(raw.get_data(), picks), raw.info)

# event_ids
event_ids = dict(ort1=2, ort2=6, ort3=9, ort4=14, ort5=17, ort6=33)

# raw issues
info = raw.info
reject = dict(grad=4000e-13)


def get_epochs(raw_object, event_id, picks=picks,
               tmin=tmin, tmax=tmax,
               baseline=(tmin, 0),
               reject=reject, preload=True):
    events = mne.find_events(raw_object)
    epochs = mne.Epochs(raw_object,
                        events=events, event_id=event_id,
                        tmin=tmin, tmax=tmax,
                        picks=picks, baseline=(tmin, 0),
                        reject=reject, preload=True)
    return epochs


def plot_evoked(epochs):
    data = epochs.get_data()
    data_mean = np.mean(data, axis=0)
    plt.plot(data_mean.transpose())
    plt.title(str(epochs.event_id))


# get_epochs(raw, event_ids).average().plot(time_unit='s', show=False)
# get_epochs(raw_custom, event_ids).average().plot(time_unit='s', show=False)

# Plot epochs
plt.figure()
plot_evoked(get_epochs(raw_custom, event_ids))

plt.figure()
idx = 1
for e in event_ids.values():
    plt.subplot(3, 2, idx)
    idx += 1
    plot_evoked(get_epochs(raw_custom, e))

plt.show()
