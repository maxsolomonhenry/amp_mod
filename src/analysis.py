from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import pyworld as pw

from defaults import ANA_PATH, DATA_PATH, EPS, PITCH_RATE
from util import (
    force_mono,
    normalize,
    read_wav,
    save_data,
    time_plot,
    trim_to_duration,
    trim_silence
)


if __name__ == '__main__':

    # Flags.
    VERBOSE = True

    # Analysis parameters.
    excerpt_in = 0.75
    excerpt_dur = 1.75

    pattern = os.path.join(ANA_PATH, '*.wav')
    audio_files = glob(pattern)
    assert audio_files, "Pattern {} yields no results.".format(pattern)

    # Calculations.
    pitch_period_ms = 1/PITCH_RATE * 1000

    for path in audio_files:
        basename = os.path.basename(path)

        if VERBOSE:
            print('Reading {}...'.format(basename))

        sr, x = read_wav(path)

        x = force_mono(x)
        x = normalize(x)
        x = trim_silence(x, threshold=-5)
        x = trim_to_duration(x, excerpt_in, excerpt_dur)

        if VERBOSE:
            print('WORLD analysis...')
        f0, sp, ap = pw.wav2world(x, sr, pitch_period_ms)

        time_plot(f0, PITCH_RATE, title=basename, show=True)