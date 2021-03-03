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
#       - FM
#       - specify vibrato rates.
#       - fixed time rather than num_cycles.

import math
import numpy as np

from analysis import single_cycles
from defaults import SAMPLE_RATE, PITCH_RATE
from util import midi_to_hz, normalize, stft_plot, upsample


class StimulusGenerator:
    """
    Generate modulating tones from a cycle of spectral envelopes.
    """
    def __init__(self, sr: int = 44100, pr: int = 200):
        assert sr > 0
        self.sr = sr

        assert pr > 0
        self.pr = pr

        self.f0 = None
        self.fm_depth = None

        self.env = None
        self.num_partials = None

        self.length = None
        self.mod_rate = None
        self.mod_hold = None
        self.mod_fade = None

    def __call__(
            self,
            f0: float,
            fm_depth: float,
            env: np.ndarray,
            num_partials: int,
            length: float,
            mod_rate: float,
            mod_hold: float,
            mod_fade: float,
    ) -> np.ndarray:

        # Argument checking.
        assert f0 > 0
        self.f0 = f0

        assert fm_depth >= 0
        self.fm_depth = fm_depth

        assert env.ndim == 2
        self.env = env

        assert num_partials > 0
        assert (num_partials * f0) <= (self.sr // 2)
        self.num_partials = num_partials

        assert length > 0
        self.length = length

        assert 0 < mod_rate <= (self.pr // 2)
        self.mod_rate = mod_rate

        assert mod_hold >= 0
        assert mod_fade >= 0
        assert (mod_hold + mod_fade) <= length
        self.mod_hold = mod_hold
        self.mod_fade = mod_fade

        # Pre-processing.
        self.resample_env()

        # Output.
        x = self.synthesize()
        return x

    def resample_env(self):
        # TODO
        pass

    def synthesize(self):
        num_samples = int(self.length * self.sr)
        x = np.zeros(num_samples)

        for k in np.arange(1, self.num_partials + 1):
            x += self.make_partial(k)
        return x

    def make_partial(self, k):
        frequency = k * self.f0
        amp_envelope = self.make_amp_envelope(frequency)
        carrier = self.make_carrier(frequency)
        return amp_envelope * carrier

    def make_amp_envelope(self, frequency):
        # TODO
        return 1.

    def make_carrier(self, frequency):
        t = np.arange(int(self.length * self.sr))/self.sr

        # Expects cycle that begins at top of cycle, i.e. cos(0).
        trajectory = np.cos(2. * np.pi * self.mod_rate * t)

        # Shape modulation.
        trajectory *= self.get_fm_coefficient()
        trajectory *= self.get_depth_trajectory()
        trajectory += 1.

        # Apply modulation.
        trajectory *= frequency

        phase = np.cumsum(2 * np.pi * trajectory / self.sr)
        return np.cos(phase)

    def get_fm_coefficient(self):
        """
        Converts `fm_depth` into coefficient for frequency trajectory.
        """
        # TODO
        return 0.01

    def get_depth_trajectory(self):
        """
        Path from 0 to 1 based on `mod_hold` and `mod_fade` times.
        """

        hold_samples = int(self.mod_hold * self.sr)
        fade_samples = int(self.mod_fade * self.sr)
        end_samples = int(self.length * self.sr - (hold_samples + fade_samples))

        hold = np.zeros(hold_samples)
        fade = np.linspace(0, 1, fade_samples, endpoint=False)
        end = np.ones(end_samples)

        return np.concatenate((hold, fade, end))


# def make_partial(
#     _frequency: float,
#     _env: np.ndarray,
#     _num_cycles: int,
# ) -> np.ndarray:
#
#     # Loop for desired number of vibrato cycles.
#     _env = np.tile(_env, [_num_cycles, 1])
#
#     total_bins = _env.shape[1]
#     tmp_frames = _env.shape[0]
#     bin_num = _frequency / (SAMPLE_RATE // 2) * total_bins
#
#     tmp_samples = math.ceil(tmp_frames * SAMPLE_RATE / PITCH_RATE)
#
#     amp_envelope = np.zeros(tmp_samples)
#     bin_frac = bin_num % 1
#
#     # Read the partial's amplitude envelope based on the desired frequency.
#     if bin_frac == 0:
#         tmp_ = loop(_env[:, int(bin_num)])
#         amp_envelope += upsample(tmp_)
#     else:
#         # Linear interpolation for fractional bin values.
#         tmp_ = loop(_env[:, math.floor(bin_num)])
#         amp_envelope += (1 - bin_frac) * upsample(tmp_)
#
#         tmp_ = loop(_env[:, math.ceil(bin_num)])
#         amp_envelope += bin_frac * upsample(tmp_)
#
#     # TODO: possibly lp filter this.
#
#     carrier = make_carrier(_frequency, amp_envelope.shape[0])
#
#     return amp_envelope * carrier


# def make_carrier(
#         frequency: float, length: int, sample_rate: int = SAMPLE_RATE
# ) -> np.ndarray:
#     t = np.arange(length) / sample_rate
#     phi = 2 * np.pi * np.random.rand()
#     return np.cos(2 * np.pi * frequency * t + phi)


def loop(in_: np.ndarray) -> np.ndarray:
    return np.concatenate([in_, [in_[0]]])


# def synthesize(_f0, _env, _num_cycles, _num_harmonics):
#     _num_frames = _num_cycles * _env.shape[0]
#     _num_samples = math.ceil(_num_frames * SAMPLE_RATE / PITCH_RATE)
#
#     x = np.zeros(_num_samples)
#
#     for k in np.arange(1, _num_harmonics):
#         x += make_partial(k*_f0, _env, _num_cycles)
#
#     return normalize(x)


def get_fm_depth(_datum):
    max_ = np.max(_datum['f0'])
    min_ = np.min(_datum['f0'])
    return max_/min_


# Synthesis parameters.
num_cycles = 10
num_partials = 70

# Midi 48 -> C3.
midi_pitch = 48

synth_out = []

for datum in single_cycles:

    fm_depth = get_fm_depth(datum)
    f0 = midi_to_hz(midi_pitch)

    # Bring to linear amplitude. Env is calculated as the power spectrum.
    env = np.sqrt(datum['env'])

    generator = StimulusGenerator(sr=SAMPLE_RATE, pr=PITCH_RATE)
    x = generator(
        f0=f0,
        fm_depth=0.01,
        env=env,
        num_partials=70,
        length=2.,
        mod_rate=5.,
        mod_hold=0.3,
        mod_fade=0.7,
    )

    x = normalize(x)

    synth_out.append(
        {
            'filename': datum['filename'],
            'f0': f0,
            'num_cycles': num_cycles,
            'wav': x,
        }
    )

    # Debugging.
    stft_plot(x)
