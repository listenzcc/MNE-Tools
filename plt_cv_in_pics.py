# coding: utf-8
# Compare raw and envlop, under cv-svm
import numpy as np
import os
import matplotlib.pyplot as plt

ort_list = [15, 45, 75, 105, 135, 165]

dir_list = ['cv_raw_lr_timeresolve', 'cv_envlop_lr_timeresolve']
fname = 'scores_store.npy'

scores_store0 = np.load(os.path.join('pics', dir_list[0], fname))
scores_store1 = np.load(os.path.join('pics', dir_list[1], fname))
epochs_times = np.load(os.path.join('pics', 'epochs_times_decim_10.npy'))


def get_idx(j, k, n=5):
    return j*n+k


fig, axes = plt.subplots(nrows=3, ncols=5)
for j, axx in zip(range(3), axes):
    for k, ax in zip(range(5), axx):
        idx = get_idx(j, k)
        # ax.set_title('%d, %d, %d' % (j, k, idx))
        scores0 = scores_store0[idx][1]
        ax.plot(epochs_times, scores0.mean(0), label='c1')
        scores1 = scores_store1[idx][1]
        ax.plot(epochs_times, scores1.mean(0), label='c2')
        ax.axhline(.5, color='k', linestyle='--', label='chance')
        ax.axvline(0, color='k')
        ort = scores_store1[idx][0]
        ax.set_title('%d, %d' % (ort_list[ort[0]], ort_list[ort[1]]))
        ax.legend()


plt.show()
