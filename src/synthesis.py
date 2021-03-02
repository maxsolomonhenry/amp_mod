import math
import numpy as np

from analysis import single_cycles
from defaults import SAMPLE_RATE, PITCH_RATE
from util import normalize, upsample


def make_partial(
    frequency: float,
    env: np.ndarray,
    num_cycles: int,
    pitch_rate: int = PITCH_RATE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:

    # Loop for desired number of vibrato cycles.
    env = np.tile(env, [num_cycles, 1])

    total_bins = env.shape[1]
    tmp_frames = env.shape[0]
    bin_num = frequency / (SAMPLE_RATE // 2) * total_bins

    tmp_samples = math.ceil(tmp_frames * SAMPLE_RATE / PITCH_RATE)

    amp_envelope = np.zeros(tmp_samples)
    bin_frac = bin_num % 1

    # Read the partial's amplitude envelope based on the desired frequency.
    if bin_frac == 0:
        tmp_ = loop(env[:, int(bin_num)])
        amp_envelope += upsample(tmp_)
    else:
        # Linear interpolation for fractional bin values.
        tmp_ = loop(env[:, math.floor(bin_num)])
        amp_envelope += (1 - bin_frac) * upsample(tmp_)

        tmp_ = loop(env[:, math.ceil(bin_num)])
        amp_envelope += bin_frac * upsample(tmp_)

    # Loop for desired number of cycles.
    # TODO: possibly lp filter this.

    carrier = make_carrier(frequency, amp_envelope.shape[0])

    return amp_envelope * carrier


def make_carrier(
        frequency: float, length: int, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    t = np.arange(length) / sample_rate
    phi = 2 * np.pi * np.random.rand()
    return np.cos(2 * np.pi * frequency * t + phi)


def loop(in_: np.ndarray) -> np.ndarray:
    return np.concatenate([in_, [in_[0]]])


# Synthesis parameters.
num_cycles = 10
num_harmonics = 70

synth_out = []

for datum in single_cycles:
    f0 = np.mean(datum['f0'])
    assert f0 * num_harmonics <= (SAMPLE_RATE / 2), 'Partials over Nyquist.'

    # Bring to linear amplitude. Env is calculated as the power spectrum.
    env = np.sqrt(datum['env'])

    num_frames = num_cycles * env.shape[0]
    num_samples = math.ceil(num_frames * SAMPLE_RATE / PITCH_RATE)

    x = np.zeros(num_samples)

    for k in np.arange(1, num_harmonics):
        x += make_partial(k*f0, env, num_cycles)

    x = normalize(x)

    synth_out.append(
        {
            'filename': datum['filename'],
            'f0': f0,
            'num_cycles': num_cycles,
            'wav': x,
        }
    )
