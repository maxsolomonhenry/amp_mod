"""
Very quick and dirty test to see if the STRF model is working properly.
"""

from src.util import read_wav
from ext.auditory import strf

demo_path = '/Users/maxsolomonhenry/Documents/Python/amp_mod/audio/syn/subject_3/block_0/BASIC_2.wav'

sr, x = read_wav(demo_path)

strf_ = strf(x, sr, duration=-1)

import matplotlib.pyplot as plt
import numpy as np

print(strf_.shape)

# Inspect frames of time-varying signal.
for i in range(0, 700, 50):
    plt.imshow(np.abs(strf_[i, 20, :, :]))
    plt.show()

# Time average before taking modulus. (Bad).
tmp = np.mean(strf_, axis=0)
plt.imshow(np.abs(tmp[20, :, :]))
plt.show()

# Time average after taking modulus. (Good).
tmp = np.mean(np.abs(strf_), axis=0)
plt.imshow(tmp[20, :, :])
plt.show()