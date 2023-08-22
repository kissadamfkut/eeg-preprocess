import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import cv2
import tempfile
import os


def general(fname):
# =============================================================================
#     fname = "X:/EEG/Felnott Kontroll/eszti220405_nemo.bdf"
# =============================================================================
    raw = mne.io.read_raw_bdf(fname, preload=True)
    raw.drop_channels(raw.ch_names[64:73])
    mont = mne.channels.make_standard_montage('biosemi64')

    raw.rename_channels({raw.ch_names[i]: mont.ch_names[i] for i in range(0, 64)})

    raw.set_montage(mont)

    raw.filter(l_freq=1, h_freq=100)

    from mne.preprocessing import(ICA, create_eog_epochs, create_ecg_epochs, corrmap)
    ica = ICA(n_components=15, max_iter='auto', random_state=97)
    ica.fit(raw)
    
    fig = ica.plot_components(show=False)
    ab = fig[0]
    ab.set_dpi(100)
    ab.set_figwidth(7.5)
    ab.set_figheight(5.7)
    
    tempfd, temppath = tempfile.mkstemp(suffix=".png", prefix="pislogas_")
    os.close(tempfd)
    
    ab.savefig(temppath)
    
    img = cv2.imread(temppath, cv2.IMREAD_COLOR)
    os.remove(temppath)
    
    #raw.plot()
    return img, raw, ica

def generalfromraw(inraw):
    raw = inraw.copy()

    raw.filter(l_freq=1, h_freq=100)

    from mne.preprocessing import(ICA, create_eog_epochs, create_ecg_epochs, corrmap)
    ica = ICA(n_components=15, max_iter='auto', random_state=97)
    ica.fit(raw)
    
    fig = ica.plot_components(show=False)
    ab = fig
    ab.set_dpi(100)
    ab.set_figwidth(7.5)
    ab.set_figheight(5.7)
    
    tempfd, temppath = tempfile.mkstemp(suffix=".png", prefix="pislogas_")
    os.close(tempfd)
    
    ab.savefig(temppath)
    
    img = cv2.imread(temppath, cv2.IMREAD_COLOR)
    os.remove(temppath)
    
    #raw.plot()
    return img, raw, ica
