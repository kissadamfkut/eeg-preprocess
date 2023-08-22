#!/bin/python
import grubbs as grubbs
import mne
from mne.preprocessing import ICA
import numpy as np
import scipy.stats
from scipy import signal
from outliers import smirnov_grubbs as grubbs
from Olivia.Autoblink import Autoblink_code
import sys
import tempfile
from nagylab import emgdetect
from matplotlib import pyplot as plt

num = 0
debug=False
debugpre=None if not debug else tempfile.mkdtemp(prefix="eegrereference_")
if debug:
    print("Using debug temp dir:", debugpre)

#n_jobs='cuda'
n_jobs=None

uri = "/psychophys/EEG/Poly/EEG/reka210325_nemo.bdf" if len(sys.argv) < 2 else sys.argv[1]
print("Using file:", uri)
raw = mne.io.read_raw_bdf(uri, preload=True)  # load bdf

raw.drop_channels(raw.ch_names[64:len(raw.ch_names)])  # Drop EMG and Signal channels

# import locations
mont = mne.channels.make_standard_montage('biosemi64')
raw.rename_channels({raw.ch_names[i]: mont.ch_names[i] for i in range(0, 64)})
raw.set_montage(mont)
del mont

print("Data loaded")

emgs = emgdetect.emgdetect(uri)
print("EMG detection done")

# calculate adjacency; 0 where [i,j] are adjucent
whip_adjacency_matrix = mne.channels.find_ch_adjacency(raw.info, 'eeg')[0]
for i in range(32):
    for j in range(32):
        whip_adjacency_matrix[i + 32, j] = 1
        whip_adjacency_matrix[i, j + 32] = 1
for i in range(64):
    for j in range(64):
        if whip_adjacency_matrix[i, j]:
            whip_adjacency_matrix[i, j] = 0
        else:
            whip_adjacency_matrix[i, j] = 1

raw.filter(l_freq=1, h_freq=100, n_jobs=n_jobs)  # bandpassfilter data from l_freq to h_freq
print("First filtering done")

gainn = 0
def gain_error(measure, base):
    global gainn
    
    measure = measure.copy()
    base = base.copy()
    
    for start, stop in emgs:
        measure[start:stop] = [0] * (stop-start)
        base[start:stop] = [0] * (stop-start)
    
    ret = scipy.stats.linregress(base, measure)
    #print(gainn // 64, gainn % 64, ret)
    #if np.abs(ret.slope - 1) > 0.3 or ret.rvalue < 0.4:
    #    print(gainn // 64, gainn % 64, ret)
    #gainn = gainn + 1
    return ret.slope

def remove_blink(inraw, stamp=""):
    ret = Autoblink_code.remove_blink(inraw)
    print("Blink components:", ret.exclude)

    if debug:
        global num
        inraw.plot(block=True).savefig(f"""{debugpre}/test_{num:02}_{"ica_preran"}_{stamp}.png""")
        num = num + 1

        ret.plot_sources(inraw, block=True).savefig(f"""{debugpre}/testica_{num:02}_ica_{"ica_components"}_{stamp}.png""")
        num = num + 1

        plotraw = inraw.copy()
        ret.apply(plotraw)
        plotraw.plot(block=True).savefig(f"""{debugpre}/test_{num:02}_{"without blink"}_{stamp}.png""")
        num = num + 1

    return ret


chs_a = range(32)
chs_b = range(32, 64)

#Channel indexes on each whip might having blink artifacts
front_back_border_a = 11
front_back_border_b = 15

front_ch_a = range(front_back_border_a+1)
back_ch_a = range(front_back_border_a+1, 32)
front_ch_b = range(32, 32 + front_back_border_b)
back_ch_b = range(32 + front_back_border_b, 64)


class cm_remover:
    def __init__(self):
        self._estimated_common_mode_a = []
        self._estimated_common_mode_b = []

    def calc_phase1(self, chs):
        self._estimated_common_mode_a = [np.mean(chs[back_ch_a], axis=0)] * 32
        self._estimated_common_mode_b = [np.mean(chs[back_ch_b], axis=0)] * 32
        return chs

    def calc_phase2(self, chs):
        self._estimated_common_mode_a = [np.mean(chs[chs_a], axis=0)] * 32
        self._estimated_common_mode_b = [np.mean(chs[chs_b], axis=0)] * 32
        return chs

    def calc_phase3(self, chs):
        cm = whip_adjacency_matrix * chs
        cm = np.multiply(cm, 1 / np.sum(whip_adjacency_matrix, axis=1))
        self._estimated_common_mode_a = [np.asarray(ch).ravel() for ch in cm[chs_a]]
        self._estimated_common_mode_b = [np.asarray(ch).ravel() for ch in cm[chs_b]]
        return chs

    def remove(self, chs):
        return np.array(
            [np.subtract(chs[i], self._estimated_common_mode_a[i]) for i in range(32)]
            +
            [np.subtract(chs[32 + i], self._estimated_common_mode_b[i]) for i in range(32)]
        )

    def remove_with_gain_compensation(self, chs):
        return np.array(
            [np.subtract(chs[i],
                         self._estimated_common_mode_a[i] * gain_error(chs[i], self._estimated_common_mode_a[i]))
             for i in range(32)]
            +
            [np.subtract(chs[32 + i],
                         self._estimated_common_mode_b[i] * gain_error(chs[32 + i], self._estimated_common_mode_b[i]))
             for i in range(32)]
        )

    def plotch(self, ch, fn):
        data=[]
        if ch < 32:
            data=self._estimated_common_mode_a[ch]
        else:
            data=self._estimated_common_mode_b[ch-32]

        plt.figure()
        plt.plot(data[0:20480])
        plt.savefig(fn)


cm_r = cm_remover()
raw_orig = raw.copy()

print("Starting the main process")
while True:
    if debug:
        raw.plot(block=True).savefig(f"""{debugpre}/test_{num:02}_{"Original"}.png""")
        num = num + 1

    raw.apply_function(cm_r.calc_phase1, ['eeg'], channel_wise=False)
    raw.apply_function(cm_r.remove, ['eeg'], channel_wise=False)

    ica = remove_blink(raw, stamp="Phase1")
    raw = raw_orig.copy()
    ica.apply(raw)
    raw.filter(l_freq=.5, h_freq=None, n_jobs=n_jobs)

    print("Entering phase2")

    for _ in range(2):
        raw.apply_function(cm_r.calc_phase2, ['eeg'], channel_wise=False)
        raw = raw_orig.copy()
        raw.apply_function(cm_r.remove_with_gain_compensation, ['eeg'], channel_wise=False)

        # remove_blink
        ica = remove_blink(raw, stamp="Phase2")
        raw = raw_orig.copy()
        ica.apply(raw)
        raw.filter(l_freq=.5, h_freq=None, n_jobs=n_jobs)

    print("Entering phase3")
    for _ in range(2):
        raw.apply_function(cm_r.calc_phase3, ['eeg'], channel_wise=False)
        raw = raw_orig.copy()
        raw.apply_function(cm_r.remove_with_gain_compensation, ['eeg'], channel_wise=False)

        # remove_blink
        ica = remove_blink(raw, stamp="Phase3")
        raw = raw_orig.copy()
        ica.apply(raw)
        raw.filter(l_freq=.5, h_freq=None, n_jobs=n_jobs)

    raw.apply_function(cm_r.remove_with_gain_compensation, ['eeg'], channel_wise=False)
    if debug:
        raw.plot(block=True).savefig(f"""{debugpre}/test_{num:02}_{"Final"}.png""")
        num = num + 1

        for ch in range(64):
            cm_r.plotch(ch, f"""{debugpre}/cm_{num:02}_{ch:02}.png""")
            num = num + 1

    #determine empty channels
    fs = raw.info.get("sfreq")
    f, Pxx_den = signal.periodogram(raw.get_data(), fs, nfft=int(2*fs), return_onesided=True, scaling='density')

    res = np.zeros(64)
    for fi in range(0, 20):
        for potential_bad_channel in grubbs.min_test_indices(np.log10(Pxx_den[:, np.abs(f - fi).argmin()])):
            res[potential_bad_channel] = res[potential_bad_channel] + 1

    new_bad_channels = [raw.ch_names[int(index)] for index in np.argwhere(res > 15)]

    if len(new_bad_channels) > 0 :
        raw_orig.info['bads'] = raw.info['bads'] + new_bad_channels
        break
    else:
        break


annotations = mne.Annotations(
        [start/int(raw.info.get("sfreq")) for start,_ in emgs],
        [(stop-start)/int(raw.info.get("sfreq")) for start,stop in emgs],
         "bad emg"
    )
raw.set_annotations(annotations)

print("Bad channels:", raw_orig.info['bads'])
print("Source:", uri)
print("Dest:")
if debug:
    raw.save(f"""{debugpre}//a_eeg.fif""")
else:
    raw.save(f"""{uri}_eeg.fif""")
