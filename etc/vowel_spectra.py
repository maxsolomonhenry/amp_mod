# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.signal import lfilter

data_path = '/Users/maxsolomonhenry/Documents/Python/vibrato_space/vibratospace/data/data.pickle'

with open(data_path, 'rb') as handle:
    data = pickle.load(handle)

spectra = {}
nfft = 8192
hN = nfft // 2 + 1


impulse = np.zeros(nfft)
impulse[0] = 1;

vc_count = 1

for datum in data:
    filename = datum['filename']
    
    a = datum['lpc']
    x = lfilter([1], a, impulse)
    X = np.fft.fft(x, nfft)
    mX = np.abs(X)[:hN]

    if filename.split('_')[0] == 'm4':
        label = os.path.splitext(filename.split('_')[3])[0]
    if filename.split('_')[0] == 'VC':
        label = 'VC{}'.format(vc_count)
        vc_count += 1

    spectra[label] = mX / np.max(mX)
# -

