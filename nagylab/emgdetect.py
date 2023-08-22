#!/bin/python
import mne
from mne.preprocessing import ICA
import numpy as np
import scipy.stats
import statistics
import json
import sys

def isIn(intervals, start, stop):
    for begin, end in intervals:
        if begin < start < end or begin < stop < end:
            return True
        
    return False

#mne.set_log_level('CRITICAL')

def emgdetect(filename):
    uri = filename
    raw = mne.io.read_raw_bdf(uri, preload=True) #load bdf
    del uri

    raw.drop_channels(raw.ch_names[64:len(raw.ch_names)])  # Drop EMG and Signal channels

    if (raw.n_times / raw.info['sfreq']) < 60:
        raise Exception("!!Too short recording")

    #import locations
    mont = mne.channels.make_standard_montage('biosemi64')
    raw.rename_channels({raw.ch_names[i]: mont.ch_names[i] for i in range(0,64)})
    raw.set_montage(mont)
    del mont

    chs_a = range(32)
    chs_b = range(32,64)

    bandl=350
    bandh=650
    raw.filter(l_freq=bandl, h_freq=bandh)

    cma = np.mean(raw.get_data()[chs_a], axis=0)
    cmb = np.mean(raw.get_data()[chs_b], axis=0)

    length = len(raw.get_data()[0,:])

    #print("Number of samples:", length)

    badness = [0] * len(raw.get_data()[0,:])
    for ch in range(64):
        if raw.ch_names[ch] in raw.info['bads']:
            continue

        #remove common mode
        if ch < 32:
            ch = raw.get_data()[ch,:] - cma
        else:
            ch = raw.get_data()[ch,:] - cmb
        
        #Calculate power
        chp = [sample ** 2 for sample in ch]
        integrate_time = 0.05 #sec
        if 1.0 / integrate_time > (bandh-bandl):
            raise Exception('integrate time lower than period time')
        power_integrate_time = int(integrate_time * raw.info['sfreq']) #0.05 sec * sample freq
        chps=scipy.signal.filtfilt([1]*power_integrate_time,1,chp)

        #Filter errors
        medfilt_width = 11 #samples
        chpsm = scipy.signal.medfilt(chps, medfilt_width)

        #Determine threshold
        snr = 15
        without_peaks = np.sort(chpsm)[power_integrate_time*2:(len(chpsm)//2)]
        midline = np.mean(without_peaks)
        thr = snr*midline

        #Decision
        above = [1 if i > thr else 0 for i in chpsm]
        abovef = scipy.signal.medfilt(above, 3)
        minimumlength = 1 # sec
        minimumlength_samples = int(raw.info['sfreq']*minimumlength)
        aboveff = scipy.signal.filtfilt([1./minimumlength_samples]*minimumlength_samples,1,abovef)
        above = [1 if i > 0 else 0 for i in aboveff]

        badness = np.add(badness, above)

    #looking for bad windows
    start_limit = 2

    sss = [0] + badness + [0]
    raiseing=0
    inwindow=False
    windows=[]
    for index in range(len(sss)):
        sample = sss[index]
        if inwindow:
            if sample == 0:
                windows.append((raiseing-1, index-1))
                inwindow=False
        else:
            if sample >= start_limit:
                raiseing=index
                inwindow=True

    fs = raw.info['sfreq']
    windows_s = [ {"start": start/fs, "stop": stop/fs} for start, stop in windows]

    return windows


