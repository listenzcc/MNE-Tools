#!/usr/bin/env python
# coding: utf-8

prefix_default = 'MEG'
suffix_default = ['1', '2', '3']

sensor_idx_default = [172, 164, 163, 184, 183, 224, 223, 244, 243, 252,
                      171, 173, 194, 191, 201, 202, 231, 232, 251, 253,
                      174, 193, 192, 204, 203, 234, 233, 254,
                      214, 212, 211, 213, 153, 263]


def num2str(n):
    return '%d' % n


def good_sensors(ch_names,
                 sensor_idx=sensor_idx_default,
                 prefix=prefix_default,
                 suffix=suffix_default):
    sensors = []
    for i in sensor_idx:
        for s in suffix:
            sensors.append(prefix+num2str(i)+s)
    picks = []
    for e in sensors:
        picks.append(ch_names.index(e))
    return sensors, picks
