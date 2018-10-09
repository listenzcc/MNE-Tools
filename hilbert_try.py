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


# File name
raw_fname = 'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTraining_1_raw_tsss.fif'

# Settings
tmin, tmax = -0.25, 1.25
freq_l, freq_h = 1, 15

# Load raw data
raw_old = mne.io.read_raw_fif(raw_fname, preload=True)
raw_old.filter(freq_l, freq_h, fir_design='firwin')
picks = mne.pick_types(raw_old.info,
                       meg=True, stim=False,
                       eeg=False, eog=False,
                       exclude='bads')

raw = mne.io.RawArray(raw_old.get_data(), raw_old.info)
raw_custom = mne.io.RawArray(get_hilbert(raw.get_data(), picks), raw.info)

# event_ids
event_ids = dict(ort1=2, ort2=6, ort3=9, ort4=14, ort5=17, ort6=33)

# raw issues
info = raw.info
reject = dict(grad=4000e-13)


def plot_raw(raw2show, picks=picks):
    events = mne.find_events(raw2show)
    epochs_raw = mne.Epochs(raw2show,
                            events=events, event_id=event_ids,
                            tmin=tmin, tmax=tmax,
                            picks=picks, baseline=(tmin, 0),
                            reject=reject, preload=True)
    evoked_raw = epochs_raw.average()
    evoked_raw.plot(time_unit='s', show=False)
    '''
    plt.figure()
    idx = 1
    for e in event_ids.values():
        epochs_tmp = mne.Epochs(raw2show,
                                events=events, event_id=e,
                                tmin=tmin, tmax=tmax,
                                picks=picks, baseline=(tmin, 0),
                                reject=reject, preload=True)
        evoked_tmp = epochs_tmp.average()
        plt.subplot(3, 2, idx)
        idx += 1
        plt.plot(evoked_tmp.data.transpose())
        plt.title(e)
    '''


plot_raw(raw_old)
plot_raw(raw)
plot_raw(raw_custom)

'''
plt.figure()
idx = 255
plt.plot(raw.get_data()[idx])
plt.plot(raw_custom.get_data()[idx])
'''


plt.show()
