import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pyworld as pw

from glob import glob
from scipy.signal import find_peaks

from defaults import ANA_PATH, DATA_PATH, EPS, PITCH_RATE
from util import (
    force_mono,
    low_pass,
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
    silence_db = -5
    pitch_lp = 10

    which_peak = 4

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
        x = trim_silence(x, threshold=silence_db)
        x = trim_to_duration(x, excerpt_in, excerpt_dur)

        if VERBOSE:
            print('WORLD analysis...')
        f0, sp, ap = pw.wav2world(x, sr, pitch_period_ms)

        # Remove first sample, which is always 0 Hz.
        tmp_f0 = f0[1:]
        tmp_f0 = low_pass(tmp_f0, pitch_lp, PITCH_RATE)

        # Add one to compensate for the subtracted index above.
        peaks = find_peaks(tmp_f0)
        peaks += 1

        # Extract one vibrato period.
        assert which_peak < len(peaks)
        if VERBOSE:
            print('Detected {} peaks. Choosing peak {}'.format(
                len(peaks), which_peak
            ))

        vib_start = peaks[which_peak]
        vib_end = peaks[which_peak + 1]

        time_plot(f0, PITCH_RATE, title=basename, show=True)