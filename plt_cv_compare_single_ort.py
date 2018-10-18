# coding: utf-8
# Compare single ort with others

import numpy as np
import os
import matplotlib.pyplot as plt

ort_list = [15, 45, 75, 105, 135, 165]

dir_target = 'cv_raw_lr_timeresolve'
fname = 'scores_store.npy'

scores_store = np.load(os.path.join('pics', dir_target, fname))
epochs_times = np.load(os.path.join('pics', 'epochs_times_decim_10.npy'))


def get_idx(j, k, n=2):
    return j*n+k


def stack(a, b):
    if len(a) == 0:
        return b
    return np.vstack((a, b))


scores_orts = []
for j in range(6):
    scores_orts.append([])

scores_diffs = []
for j in range(6):
    scores_diffs.append([])

nrows, ncols = 3, 5
fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
for j, axx in zip(range(nrows), axes):
    for k, ax in zip(range(ncols), axx):
        idx = get_idx(j, k, ncols)
        ax.set_title('%d, %d, %d' % (j, k, idx))
        pare = scores_store[idx][0]
        scores = scores_store[idx][1]
        ax.plot(epochs_times, scores.mean(0), label='roc')
        ax.axhline(.5, color='k', linestyle='--', label='chance')
        ax.axvline(0, color='k')
        ax.set_title('%d, %d' % (ort_list[pare[0]], ort_list[pare[1]]))
        ax.legend()

        for m in range(2):
            scores_orts[pare[m]] = stack(scores_orts[pare[m]], scores)

        d = pare[1]-pare[0]
        scores_diffs[d] = stack(scores_diffs[d], scores)


nrows, ncols = 2, 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
for j, axx in zip(range(nrows), axes):
    for k, ax in zip(range(ncols), axx):
        idx = get_idx(j, k, ncols)
        ax.set_title('%d, %d, %d' % (j, k, idx))
        scores = scores_orts[idx]
        ax.plot(epochs_times, scores.mean(0), label='roc')
        ax.axhline(.5, color='k', linestyle='--', label='chance')
        ax.axvline(0, color='k')
        ax.set_title('%d vs others' % ort_list[idx])
        ax.legend()


nrows, ncols = 2, 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
for j, axx in zip(range(nrows), axes):
    for k, ax in zip(range(ncols), axx):
        idx = get_idx(j, k, ncols)
        ax.set_title('%d, %d, %d' % (j, k, idx))
        scores = scores_diffs[idx]
        if len(scores) == 0:
            continue
        ax.plot(epochs_times, scores.mean(0), label='roc')
        ax.axhline(.5, color='k', linestyle='--', label='chance')
        ax.axvline(0, color='k')
        ax.set_title('%d diff' % (idx*30))
        ax.legend()


plt.show()
