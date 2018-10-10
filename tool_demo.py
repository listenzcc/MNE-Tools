# coding: utf-8

import mne
import matplotlib.pyplot as plt
from mnetools_zcc import prepare_raw, get_envlop, get_epochs, plot_evoked


fname = 'D:\\BeidaShuju\\rawdata\\QYJ\\MultiTraining_1_raw_tsss.fif'

event_ids = dict(ort015=2,  ort045=6,  ort075=9,
                 ort105=14, ort135=17, ort165=33)

raw, picks = prepare_raw(fname)

freq_l, freq_h = 1, 15
raw.filter(freq_l, freq_h, fir_design='firwin')

raw_custom = mne.io.RawArray(get_envlop(raw.get_data(), picks), raw.info)

epochs = get_epochs(raw_custom, event_ids, picks)

plt.figure()
plot_evoked(epochs)

plt.show()
