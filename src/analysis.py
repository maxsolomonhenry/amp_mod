# TODO  - min/max per partials.

import os
import pyworld as pw

from glob import glob
from scipy.signal import find_peaks, decimate

from defaults import ANA_PATH, PITCH_RATE
from util import (
    force_mono,
    get_amp_envelope,
    low_pass,
    normalize,
    read_wav,
    trim_to_duration,
    trim_silence
)

# Flags.
VERBOSE = False

# Analysis parameters.
excerpt_in = 0.75
excerpt_dur = 1.75
silence_db = -5
pitch_lp = 10
which_peak = 4
amp_env_cutoff = 25.

pattern = os.path.join(ANA_PATH, '*.wav')
audio_files = glob(pattern)
assert audio_files, "Pattern {} yields no results.".format(pattern)

# Calculations.
pitch_period_ms = 1/PITCH_RATE * 1000

single_cycles = []

for path in audio_files:
    basename = os.path.basename(path)

    if VERBOSE:
        print('Reading {}...'.format(basename))

    sr, x = read_wav(path)

    x = force_mono(x)
    x = normalize(x)
    x = trim_silence(x, threshold=silence_db, sr=sr)
    x = trim_to_duration(x, excerpt_in, excerpt_dur)

    if VERBOSE:
        print('WORLD analysis...')

    _f0, t = pw.dio(x, sr, frame_period=pitch_period_ms)
    f0 = pw.stonemask(x, _f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)

    # Remove the first sample, which is always 0 Hz.
    tmp_f0 = f0[1:]
    tmp_f0 = low_pass(tmp_f0, pitch_lp, PITCH_RATE)

    # Add `1` to compensate for the subtracted index above.
    peaks = find_peaks(tmp_f0)[0]
    peaks += 1

    # Extract one vibrato period.
    assert which_peak < len(peaks)
    if VERBOSE:
        print('Detected {} peaks. Choosing peak {}.'.format(
            len(peaks), which_peak
        ))

    start = peaks[which_peak]
    end = peaks[which_peak + 1]

    start_sr = int(round(start / PITCH_RATE * sr))
    end_sr = int(round(end / PITCH_RATE * sr))

    # TODO
    # amp_env = get_amp_envelope(x, cutoff=amp_env_cutoff, sr=sr)
    # amp_env = amp_env[start_sr:end_sr]

    single_cycles.append(
        {
            'filename': basename,
            'env': sp[start:end, :],
            'f0': f0[start:end],
            'sr': sr,
        }
    )
