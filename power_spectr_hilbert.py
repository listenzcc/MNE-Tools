# coding: utf-8

import mne
from mne.time_frequency import tfr_morlet
import numpy as np
import matplotlib.pyplot as plt
from mnetools_zcc import prepare_raw, get_envlop, get_epochs


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

freq_l, freq_h = 1, 15

j = 0
fname = fname_list[j]
print(fname)
raw, picks = prepare_raw(fname)
raw.filter(freq_l, freq_h, fir_design='firwin')

raw_custom = mne.io.RawArray(raw.get_data(), raw.info)
epochs_custom = get_epochs(raw_custom, event_ids, picks)

raw_hilbert = mne.io.RawArray(get_envlop(raw.get_data(), picks), raw.info)
epochs_hilbert = get_epochs(raw_hilbert, event_ids, picks)

freqs = np.logspace(*np.log10([1, 16]), num=20)
n_cycles = freqs / 2.  # different number of cycle per frequency
tmin, tmax = -0.25, 1.25


def plt_power(epochs_to_plt, timefreqs,
              picks=picks, freqs=freqs, n_cycles=n_cycles,
              tmin=tmin, tmax=tmax):

    power, itc = tfr_morlet(epochs_to_plt, freqs=freqs,
                            n_cycles=n_cycles, use_fft=True,
                            return_itc=True,
                            decim=1, n_jobs=1)
    power.plot_joint(baseline=(tmin, 0), show=False,
                     mode='mean',
                     tmin=tmin, tmax=tmax,
                     timefreqs=timefreqs)


plt_power(epochs_custom,
          timefreqs=[(0.8, 9.5), (1.0, 9.5), (0.2, 6)])
plt_power(epochs_hilbert,
          timefreqs=[(0.2, 2.1), (0.8, 9.5), (1.0, 9.5)])

plt.show()
