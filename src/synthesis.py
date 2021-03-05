"""
The BASIC stimulus has both FM and the complex AM derived from scanning the
resonant structure of the instrument, as described by Mathews and others
(Mathews and Kohut, 1973), and serves as a positive control.

A FROZEN condition keeps the complex AM of a plausible resonant structure, but
contains no FM.

In a random AM phase (RAP) condition, each partial is amplitude
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

from copy import copy
import math
import numpy as np

from defaults import SAMPLE_RATE, PITCH_RATE
from util import midi_to_hz, normalize, stft_plot, resample


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

        self.frame_rate = None

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
        """Generate a spectral- and frequency- modulated tone.

        Args:
            f0: Fundamental pitch of the output, in Hz.
            fm_depth: Depth of pitch modulation, in semitones.
            env: Array of spectral envelopes (time x real frequency).
            num_partials: Number of partials for resynthesis.
            length: Synthesis length in seconds.
            mod_rate: Rate of spectral- and frequency- modulation, in Hz.
            mod_hold: Time before applying modulation, in seconds.
            mod_fade: Time to ramp modulation (from 0 to 1), in seconds.

        Returns:
            Numpy array. A normalized, one-dimensional, audio rate stimulus.
        """

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

        # Preliminary calculations.
        num_frames = env.shape[0]
        self.frame_rate = num_frames * self.mod_rate

        # Pre-processing.
        self.process_env()

        # Output.
        x = self.synthesize()
        x = normalize(x)
        return x

    def process_env(self):

        # Preliminaries.
        num_samples = self.get_num_samples()
        num_cycles = math.ceil(self.length * self.mod_rate)

        # Extend in time to desired output length.
        tmp_env = self.env
        tmp_env = np.tile(tmp_env, [num_cycles, 1])

        # Wrap-around first value to extend interpolation.
        tmp_env = self.loop(tmp_env)

        # Resample and truncate.
        tmp_env = self._resample(tmp_env)
        tmp_env = tmp_env[:num_samples, :]

        tmp_env = self.apply_spectral_fade(tmp_env)

        self.env = tmp_env

    def apply_spectral_fade(self, tmp_env):
        """
        Fade in spectral modulation, taking the middle of the cycle as neutral.
        """
        fade = self.get_depth_trajectory()

        # Retrieve spectrum from middle of cycle.
        mid_cycle_index = round(self.env.shape[0] // 2)
        mid_env = self.env[mid_cycle_index, :]

        # Fade out middle spectrum.
        mid_env = np.outer((1 - fade), mid_env)

        # Fade in spectral modulation.
        tmp_env = tmp_env * fade[:, None]

        # Cross-fade.
        tmp_env += mid_env

        return tmp_env

    def _resample(self, tmp_env):
        return resample(tmp_env, self.frame_rate, self.sr)

    def synthesize(self):
        num_samples = self.get_num_samples()
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
        # Find the (possibly fractional) bin corresponding to `frequency`.
        bin_num = frequency / (SAMPLE_RATE // 2) * self.env.shape[1]
        bin_fraction = bin_num % 1

        num_samples = self.get_num_samples()
        amp_envelope = np.zeros(num_samples)

        # Read amplitude envelope based on the desired frequency.
        if bin_fraction == 0:
            amp_envelope += self.env[:, bin_num]
        else:
            # Linear interpolation between adjacent bins.
            amp_envelope += (1 - bin_fraction) * self.env[:, math.floor(bin_num)]
            amp_envelope += bin_fraction * self.env[:, math.ceil(bin_num)]

        # TODO possibly LP filter this to `self.frame_rate//2`

        return amp_envelope

    def make_carrier(self, frequency):
        t = np.arange(int(self.length * self.sr))/self.sr

        # Always begins at top of cycle (i.e. cos(0)), as per analysis.py.
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
        Converts `fm_depth` from semitones into coefficient for frequency.

        Note: strictly speaking, pitch modulation should be applied in
        log-space, because pitch scales logarithmically with frequency. Here, we
        apply modulation on a linear scale. Tsk tsk. For the small modulation
        excursions associated with typical vibrato, the difference is quite
        minimal, and arguably imperceptible. Though that, *sigh*, would be
        another experiment.
        """
        return 2 ** (self.fm_depth / 12) - 1

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

    def get_num_samples(self):
        return int(self.length * self.sr)

    @staticmethod
    def loop(in_: np.ndarray) -> np.ndarray:
        return np.concatenate([in_, [in_[0]]])


class EnvelopeMorpher:
    """
    Generate variations of spectral modulation based on a prototype cycle.
    """
    # TODO: log morphs/stats.
    def __init__(self, env: np.ndarray):
        assert env.ndim == 2
        self.env = copy(env)

    def shuffle_phase(self, num_shifts: int = 4):
        """
        Randomly shuffle each column.
        """

        all_shifts = np.linspace(0, 1, num_shifts, endpoint=False)
        num_frames, num_bins = self.env.shape

        for k in np.arange(num_bins):
            shift = np.random.choice(all_shifts)

            tmp = self.env[:, k]
            tmp = self.roll(tmp, shift)

            self.env[:, k] = tmp

    def __call__(self):
        return self.env

    @staticmethod
    def roll(in_, shift):
        """
        Circular shift array using linear interpolation, where 0 <= `shift` < 1
        """
        num_samples = in_.size
        shift_samples = num_samples * shift
        shift_fraction = shift_samples % 1

        out_ = np.zeros(num_samples)

        if shift_samples == 0:
            out_ += in_
        elif shift_fraction == 0:
            out_ += np.roll(in_, shift_samples)
        else:
            out_ += (1 - shift_fraction) * np.roll(in_, math.floor(shift_samples))
            out_ += shift_fraction * np.roll(in_, math.ceil(shift_samples))

        return out_


if __name__ == '__main__':
    from analysis import single_cycles

    # Helper.
    def get_fm_depth(_datum):
        """
        FM depth in semitones, calculated from middle to extrema of pitch.
        """
        max_ = np.max(_datum['f0'])
        min_ = np.min(_datum['f0'])
        return 12 * np.log2(max_/min_) / 2

    # Synthesis parameters.
    num_partials = 70

    # Midi 48 -> C3.
    midi_pitch = 48

    synth_out = []

    for datum in single_cycles:
        fm_depth = get_fm_depth(datum)
        f0 = midi_to_hz(midi_pitch)

        # Bring to linear amplitude. Env is calculated as the power spectrum.
        env_ = np.sqrt(datum['env'])

        morpher = EnvelopeMorpher(env_)
        morpher.shuffle_phase(num_shifts=4)

        generator = StimulusGenerator(sr=SAMPLE_RATE, pr=PITCH_RATE)
        x = generator(
            f0=f0,
            fm_depth=0.0,
            env=morpher(),
            num_partials=70,
            length=2.1,
            mod_rate=5.,
            mod_hold=0.3,
            mod_fade=0.7,
        )

        stft_plot(x)

        synth_out.append(
            {
                'filename': datum['filename'],
                'f0': f0,
                'wav': x,
            }
        )
