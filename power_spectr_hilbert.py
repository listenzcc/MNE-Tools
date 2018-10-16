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

event_ids_all = dict(ort015=2,  ort045=6,  ort075=9,
                     ort105=14, ort135=17, ort165=33)

event_ids = dict(ort1=2, ort2=33)

freq_l, freq_h = 1, 15

j = 0
fname = fname_list[j]
print(fname)
raw, picks = prepare_raw(fname)
raw.filter(freq_l, freq_h, fir_design='firwin')

raw_custom = mne.io.RawArray(raw.get_data(), raw.info)

raw_hilbert = mne.io.RawArray(get_envlop(raw.get_data(), picks), raw.info)

freqs = np.logspace(*np.log10([1, 16]), num=20)
n_cycles = freqs / 2.  # different number of cycle per frequency
tmin, tmax = -0.25, 1.25


def plt_power(raw_to_plt, timefreqs, event_ids, picks,
              freqs=freqs, n_cycles=n_cycles,
              tmin=tmin, tmax=tmax, plot=True):
    # plot or return power spectra from raw_to_plt
    epochs = get_epochs(raw_to_plt, event_ids, picks)
    # return itc to pervent power is a truple
    power, itc = tfr_morlet(epochs, freqs=freqs,
                            n_cycles=n_cycles, use_fft=True,
                            return_itc=True,
                            decim=10, n_jobs=6)
    if not(plot):
        # return power if do not plot
        return power
    power.plot_joint(baseline=(tmin, 0), show=False,
                     mode='mean',
                     tmin=tmin, tmax=tmax,
                     timefreqs=timefreqs)
    return power


plt.close()

power_custom = plt_power(raw_custom, plot=False,
                         event_ids=event_ids, picks=picks,
                         timefreqs=[(0.8, 9.5), (1.0, 9.5), (0.2, 6)])
power_hilbert = plt_power(raw_hilbert, plot=False,
                          event_ids=event_ids, picks=picks,
                          timefreqs=[(0.2, 2.1), (0.8, 9.5), (1.0, 9.5)])

sz = power_custom.data.shape
x_ind = np.arange(sz[2])
x_lab = np.arange(tmin, tmax+(tmax-tmin)/(sz[2]-1),
                  (tmax-tmin)/(sz[2]-1))
y_ind = np.arange(sz[1])
y_lab = freqs


def imshow_power(ax, data,
                 x_ind=x_ind, x_lab=x_lab,
                 y_ind=y_ind, y_lab=y_lab):
    # plot power spectra in simple, controllable mode
    ax.imshow(np.mean(data, axis=0))
    ax.set_xticks(x_ind[0:-1:20])
    ax.set_xticklabels(
        list('%.2f' % e for e in x_lab[0:-1:20]))
    ax.set_yticks(y_ind[0:-1:2])
    ax.set_yticklabels(
        list('%.2f' % e for e in y_lab[0:-1:2]))


# Prepare 3x2 subplots to plot power spectra
fig, axs = plt.subplots(nrows=3, ncols=2)
for ax, ort in zip(axs.flat, event_ids_all.items()):
    print(ort)
    event_id = dict()
    event_id[ort[0]] = ort[1]
    power = plt_power(raw_custom, plot=False,
                      event_ids=event_id, picks=picks,
                      timefreqs=[(0.8, 9.5), (1.0, 9.5), (0.2, 6)])
    imshow_power(ax, power.data)
    ax.set_title(ort)
plt.savefig('./pics/power_spectra_raw.png')

# Prepare 3x2 subplots to plot power spectra
fig, axs = plt.subplots(nrows=3, ncols=2)
for ax, ort in zip(axs.flat, event_ids_all.items()):
    print(ort)
    event_id = dict()
    event_id[ort[0]] = ort[1]
    power = plt_power(raw_hilbert, plot=False,
                      event_ids=event_id, picks=picks,
                      timefreqs=[(0.8, 9.5), (1.0, 9.5), (0.2, 6)])
    imshow_power(ax, power.data)
    ax.set_title(ort)
plt.savefig('./pics/power_spectra_hilbert.png')

plt.show()
