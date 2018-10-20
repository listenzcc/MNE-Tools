# coding: utf-8
# Compare raw and envlop, under cv-svm
import numpy as np
import os
import matplotlib.pyplot as plt

ort_list = [15, 45, 75, 105, 135, 165]

dir_list = ['QYJcv10_env_pca_lrl1',
            'QYJcv10_raw_pca_lrl1',
            'QYJcv10_rawenv_pca_lrl1',
            'QYJcv10_rawenv_pca30_lrl1']
score_fname = 'scores_store.npy'
times_fname = 'epochs_times.npy'

score_list = list(np.load(os.path.join('pics', e, score_fname))
                  for e in dir_list)
times_list = list(np.load(os.path.join('pics', e, times_fname))
                  for e in dir_list)


def get_idx(j, k, n=5):
    return j*n+k


fig, axes = plt.subplots(nrows=3, ncols=5)
for j, axx in zip(range(3), axes):
    for k, ax in zip(range(5), axx):
        idx = get_idx(j, k, n=5)
        # ax.set_title('%d, %d, %d' % (j, k, idx))
        for n in range(len(dir_list)):
            score2plt = np.mean(score_list[n][idx][1], 0)
            times2plt = times_list[n]
            ax.plot(times2plt, score2plt, label=dir_list[n])
        ax.axhline(.5, color='k', linestyle='--', label='chance')
        ax.axvline(0, color='k')
        ort = score_list[0][idx][0]
        ax.set_title('%d, %d' % (ort_list[ort[0]], ort_list[ort[1]]))
        ax.legend()


plt.show()
