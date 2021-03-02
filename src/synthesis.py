"""
The BASIC stimulus has both FM and the complex AM derived from scanning the
resonant structure of the instrument, as described by Mathews and others
(Mathews and Kohut, 1973), and serves as a positive control.

A FROZEN condition keeps the complex AM of a plausible resonant structure, but
contains no FM. In a random AM phase (RAP) condition, each partial is amplitude
modulated at the vibrato rate, having a random gain between 0 and 10 dB, and
with an initial modulation phase randomly selected as one of four values equal
divisions of the oscillation: {0, π/2, π, 3π/2}.

In the random AM frequency (RAF) condition, each partial is amplitude modulated
at a frequency selected from a random exponential distribution between 4 Hz and
12 Hz, the span of commonly performed vibrato rates (Verfaille, 2005). Each
partial has a random modulation gain between 0 and 10 dB.

In the pure AM condition (PAM) the entire amplitude envelope is modulated by
10 dB at the vibrato rate.

A negative control is included that has no AM or FM.

In all non-BASIC conditions, the partial gains are set to a "resting" value
determined by averaging the time-varying spectral envelope of a single vibrato
cycle. Only the BASIC condition has FM. All stimuli are generated at two vibrato
speeds, 5 Hz reflecting typical vibrato, and 2 Hz, reflecting an unrealistically
slow vibrato. Parameters for all random stimuli are logged.
"""

# TODO  - fade for modulations.
#       - incorporate FM.
#       - for fixed time rather than num_cycles.

import math
import numpy as np

from analysis import single_cycles
from defaults import SAMPLE_RATE, PITCH_RATE
from util import midi_to_hz, normalize, upsample


def make_partial(
    _frequency: float,
    _env: np.ndarray,
    _num_cycles: int,
) -> np.ndarray:

    # Loop for desired number of vibrato cycles.
    _env = np.tile(_env, [_num_cycles, 1])

    total_bins = _env.shape[1]
    tmp_frames = _env.shape[0]
    bin_num = _frequency / (SAMPLE_RATE // 2) * total_bins

    tmp_samples = math.ceil(tmp_frames * SAMPLE_RATE / PITCH_RATE)

    amp_envelope = np.zeros(tmp_samples)
    bin_frac = bin_num % 1

    # Read the partial's amplitude envelope based on the desired frequency.
    if bin_frac == 0:
        tmp_ = loop(_env[:, int(bin_num)])
        amp_envelope += upsample(tmp_)
    else:
        # Linear interpolation for fractional bin values.
        tmp_ = loop(_env[:, math.floor(bin_num)])
        amp_envelope += (1 - bin_frac) * upsample(tmp_)

        tmp_ = loop(_env[:, math.ceil(bin_num)])
        amp_envelope += bin_frac * upsample(tmp_)

    # TODO: possibly lp filter this.

    carrier = make_carrier(_frequency, amp_envelope.shape[0])

    return amp_envelope * carrier


def make_carrier(
        frequency: float, length: int, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    t = np.arange(length) / sample_rate
    phi = 2 * np.pi * np.random.rand()
    return np.cos(2 * np.pi * frequency * t + phi)


def loop(in_: np.ndarray) -> np.ndarray:
    return np.concatenate([in_, [in_[0]]])


def synthesize(_f0, _env, _num_cycles, _num_harmonics):
    _num_frames = _num_cycles * _env.shape[0]
    _num_samples = math.ceil(_num_frames * SAMPLE_RATE / PITCH_RATE)

    x = np.zeros(_num_samples)

    for k in np.arange(1, _num_harmonics):
        x += make_partial(k*_f0, _env, _num_cycles)

    return normalize(x)


def get_fm_depth(_datum):
    max_ = np.max(_datum['f0'])
    min_ = np.min(_datum['f0'])
    return max_/min_

# Synthesis parameters.
num_cycles = 10
num_harmonics = 70

# Midi 48 -> C3.
midi_pitch = 48

synth_out = []

for datum in single_cycles:

    fm_depth = get_fm_depth(datum)

    f0 = midi_to_hz(midi_pitch)
    assert f0 * num_harmonics <= (SAMPLE_RATE / 2), 'Partials over Nyquist.'

    # Bring to linear amplitude. Env is calculated as the power spectrum.
    env = np.sqrt(datum['env'])

    x = synthesize(f0, env, num_cycles, num_harmonics)

    synth_out.append(
        {
            'filename': datum['filename'],
            'f0': f0,
            'num_cycles': num_cycles,
            'wav': x,
        }
    )
