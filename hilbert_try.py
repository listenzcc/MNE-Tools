# coding: utf-8

import mne
import numpy as np
import matplotlib.pyplot as plt


# File name
raw_fname = 'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTraining_1_raw_tsss.fif'

# Settings
tmin, tmax = -0.25, 1.25
freq_l, freq_h = 1, 15

# Load raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# Filter raw data
raw.filter(freq_l, freq_h, fir_design='firwin')

# Setting sensors to pick
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
                       exclude='bads')

# Prepare issues for 'EpochsArray'
info = raw.info
events = mne.find_events(raw)
event_ids = dict(ort1=2, ort2=6, ort3=9, ort4=14, ort5=17, ort6=33)
reject = dict(grad=4000e-13)

# Pick data
data = raw.get_data()[picks]
idx = raw.ch_names.index('MEG1941')
print('idx is %d' % idx)
data_pick = data[idx]
timeline = data_pick*0
timeline[events[0:-2, 0]] = 1.1*max(data_pick)
plt.plot(data_pick)
plt.plot(timeline)
plt.plot(-timeline)
plt.title(raw.ch_names[idx])
plt.show()
