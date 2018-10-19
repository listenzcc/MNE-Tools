# coding: utf-8

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack


def analysis_hilbert(x):
    # Calculate hilbert analytical signal
    y = fftpack.hilbert(x)
    return np.sqrt(x**2 + y**2)


def get_envlop(data, picks):
    # Calculate envlop of high freq signal
    # Using hilbert analytical signal
    # picks: only transform signals, not markers
    for j in picks:
        # Calculate envlop of each sensor
        data[j] = analysis_hilbert(data[j])
    return data


reject = dict(mag=5e-12, grad=4000e-13)


def prepare_raw(fname, meg=True,
                eeg=False, eog=False, stim=False,
                reject=reject, exclude='bads'):
    # Prepare raw object
    raw = mne.io.read_raw_fif(fname, preload=True)
    # Reform raw into RawArray is necessary
    raw = mne.io.RawArray(raw.get_data(), raw.info)
    # Get picks
    picks = mne.pick_types(raw.info, meg=meg,
                           eeg=eeg, eog=eog, stim=stim,
                           exclude='bads')
    return raw, picks


tmin, tmax = -0.25, 1.25
decim = 10


def get_epochs(raw_object, event_id, picks,
               tmin=tmin, tmax=tmax,
               baseline=(tmin, 0), decim=decim,
               reject=reject, preload=True):
    # Find events
    events = mne.find_events(raw_object)
    # Get epochs on events
    epochs = mne.Epochs(raw_object, decim=decim,
                        events=events, event_id=event_id,
                        tmin=tmin, tmax=tmax,
                        picks=picks, baseline=(tmin, 0),
                        reject=reject, preload=preload)
    return epochs


def plot_evoked(epochs):
    # Plot colorful epochs
    data = epochs.get_data()
    data_mean = np.mean(data, axis=0)
    plt.plot(data_mean.transpose())
    plt.title(str(epochs.event_id))


def stack_3Ddata(a, b):
    # Stack 3D data on first dimension
    if len(a) == 0:
        return b
    sa = a.shape
    sb = b.shape
    return np.vstack((a.reshape(sa[0], sa[1]*sa[2]),
                      b.reshape(sb[0], sb[1]*sb[2]))).reshape(sa[0]+sb[0], sa[1], sa[2])
