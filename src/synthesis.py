"""
Stimuli can take on one of eight conditions:

The BASIC stimulus has both FM and the complex AM derived from scanning the
resonant structure of the instrument, as described by Mathews and others
(Mathews and Kohut, 1973), and serves as a positive control.

A FROZEN condition keeps the complex AM of a plausible resonant structure, but
contains no FM.

A SHUFFLE condition keeps the AM trajectory of each partial, but shuffles
the phase of each by a random quarter of the cycle: {0, π/2, π, 3π/2}.

The SIMPLE condition keeps the average spectral envelope and partial-wise
modulation depths of the original signal, but replaces the modulator signals
with one-cycle sinusoids all having the same modulation rate. Each partial takes
the time-averaged spectral envelope for its center amplitude. Partials are
amplitude-modulated such that their peak- and minimum- gains match the original
time-varying spectral envelope. The phase of these modulators is necessarily
randomized.

The random gain (RAG) condition is constructed much like the SIMPLE condition,
except its modulation depths are randomly determined per-partial to a value
between 0 – 10 dB according to the following formula:

    G_k = 20\log_{10}\left(\frac{\max p_k(t)}{\min p_k(t)}\right)

where Gk is the modulation depth of the kth partial, and pk(t) is the partial
amplitude. Random resonance gain has been used to construct compelling models of
vibrato in the past (Gough, 2005); though here by scrambling both partial phase
and gain, this condition effectively avoids implying any plausible resonant
structure.

The SHUFFLE, SIMPLE, and RAG conditions are also alternately synthesized in a
random AM frequency (RAF) condition. Here, each partial is amplitude modulated
at a frequency selected from a random exponential distribution between 4 Hz and
7 Hz, the span of commonly performed vibrato rates (Sundberg, 1987). We choose
an exponential distribution rather than a linear one, based on existing evidence
suggesting an exponential perceptual scaling of modulation rates (Grant, 1998).
Note that in this condition, the relative phase of the modulators is implicitly
randomized.

A PAM condition has only one global amplitude modulation. It sums the amplitudes
of the partials together, then applies it to a time-averaged spectral envelope.

An FM-ONLY stimulus applies FM to a time-averaged spectral envelope.

A CONTROL stimulus is included that has no AM or FM.

In all non-BASIC conditions, the partial gains are set to a "resting" value
measured from the time-varying spectral envelope at one quarter of a single
vibrato cycle, corresponding to the "middle" of the vibrato cycle. Only the
BASIC condition has FM. Parameters for all random stimuli are logged.

"""

from copy import copy
import math
import numpy as np

from defaults import EPS, SAMPLE_RATE, PITCH_RATE
from util import (
    add_fade, midi_to_hz, normalize, plot_envelope, stft_plot, remove_dc,
    resample
)


class StimulusGenerator:
    """
    Generate modulating tones from a cycle of spectral envelopes.
    """
    def __init__(
            self,
            sr: int = SAMPLE_RATE,
            pr: int = PITCH_RATE,
            random_rate_upper_limit: float = 12.,
            random_rate_lower_limit: float = 4.,
    ):
        assert sr > 0
        self.sr = sr

        assert pr > 0
        self.pr = pr

        assert random_rate_upper_limit >= 0
        self.random_rate_upper_limit = random_rate_upper_limit

        assert random_rate_lower_limit <= random_rate_upper_limit
        self.random_rate_lower_limit = random_rate_lower_limit

        self.f0 = None
        self.fm_depth = None

        self.env = None
        self.num_partials = None

        self.length = None
        self.mod_rate = None
        self.mod_hold = None
        self.mod_fade = None

        self.synth_mode = None
        self.audio_fade = None

        self.processed_env = None

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
            synth_mode: str = 'default',
            audio_fade: float = 0.,
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
            synth_mode: Synthesis type ->
                'default' is normal behaviour.
                'pam' is the Pure Amplitude Modulation condition (tremolo).
                'raf' is the Random Amplitude modulation Frequency condition.

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

        assert synth_mode in ['default', 'pam', 'raf']
        self.synth_mode = synth_mode

        assert 0 <= audio_fade <= length

        # Resample, loop and extend spectral envelope.
        self.process_env()

        # Output.
        x = self.synthesize()
        x = remove_dc(x)
        x = normalize(x)

        # Fade in/out.
        x = add_fade(x, audio_fade, self.sr)
        x = add_fade(x, audio_fade, self.sr, fade_out=True)
        return x

    def process_env(self):

        if self.synth_mode == 'raf':
            tmp_env = self.get_raf_env()
        else:
            tmp_env = self.cycle_and_resample_env()

        tmp_env = self.apply_spectral_fade(tmp_env)

        self.processed_env = tmp_env

    def get_raf_env(self):
        """
        Generate envelope with each partial amp-modulated at a different rate.
        """
        num_frames = self.env.shape[0]
        num_partials = self.num_partials

        num_samples = self.get_num_samples()
        out_ = np.zeros([num_samples, num_partials])

        tmp_env = copy(self.env)

        for k in range(num_partials):
            frequency = (k + 1) * self.f0

            random_rate = self.get_random_rate()
            num_cycles = math.ceil(self.length * random_rate)
            frame_rate = num_frames * random_rate

            # Calculate partial trajectory.
            tmp = self.get_amp_from_frequency(frequency, tmp_env)

            # Cycle to desired synthesis length and resample.
            tmp = np.tile(tmp, num_cycles)
            tmp = self.loop(tmp)
            tmp = self._resample(tmp, frame_rate)

            # Truncate and place in output.
            out_[:, k] = tmp[:num_samples]

        return out_

    def get_random_rate(self):
        upper = self.random_rate_upper_limit
        lower = self.random_rate_lower_limit
        x = np.random.rand()
        return (upper - lower + 1.)**x + lower - 1.

    def cycle_and_resample_env(self):

        # Preliminaries.
        num_frames = self.env.shape[0]
        num_cycles = math.ceil(self.length * self.mod_rate)
        num_samples = self.get_num_samples()

        # Calculate only required harmonic partials.
        tmp_env = copy(self.env)
        tmp_env = self.reduce_to_relevant_partials(tmp_env)

        # Extend in time to desired output length.
        tmp_env = np.tile(tmp_env, [num_cycles, 1])

        # Wrap-around first value to extend interpolation.
        tmp_env = self.loop(tmp_env)

        # Resample.
        frame_rate = num_frames * self.mod_rate
        tmp_env = self._resample(tmp_env, frame_rate)

        # Truncate.
        tmp_env = tmp_env[:num_samples, :]

        return tmp_env

    def reduce_to_relevant_partials(self, tmp_env):
        """
        Extract only spectral information relevant to synthesis.
        """

        num_frames = tmp_env.shape[0]
        num_partials = self.num_partials

        out_ = np.zeros([num_frames, num_partials])

        for k in range(self.num_partials):
            frequency = (k + 1) * self.f0
            out_[:, k] += self.get_amp_from_frequency(frequency, tmp_env)

        return out_

    def get_amp_from_frequency(self, frequency, tmp_env):
        """
        Linear interpolate to extract frequency-wise amplitude envelope.
        """

        # Expand to facilitate broadcasting if necessary (code for 1- or 2-d).
        if tmp_env.ndim == 1:
            tmp_env = tmp_env[None, :]

        num_frames = tmp_env.shape[0]
        amp_envelope = np.zeros(num_frames)

        # Find the (possibly fractional) bin corresponding to `frequency`.
        bin_num = self.get_bin_num(frequency)
        bin_fraction = bin_num % 1

        bin_floor = math.floor(bin_num)
        bin_ceil = math.ceil(bin_num)

        # Read amplitude envelope based on the desired frequency.
        if bin_fraction == 0:
            amp_envelope += tmp_env[:, bin_num]
        else:
            # Linear interpolation between adjacent bins.
            amp_envelope += (1 - bin_fraction) * tmp_env[:, bin_floor]
            amp_envelope += bin_fraction * tmp_env[:, bin_ceil]

        return amp_envelope

    def apply_spectral_fade(self, tmp_env):
        """
        Fade in spectral modulation, taking approx pi/2 of the cycle as neutral.
        """

        fade = self.get_depth_trajectory()

        mid_env = self.get_mid_env()

        # Fade out middle spectrum.
        mid_env = np.outer((1 - fade), mid_env)

        # Fade in spectral modulation.
        tmp_env = tmp_env * fade[:, None]

        # Cross-fade.
        tmp_env += mid_env

        return tmp_env

    def get_mid_env(self):
        """Retrieve spectrum from 1/4 of cycle.

        This is approximately pi/2 or 3pi/2, i.e. where the vibrato trajectory
        is in the middle of its throw (not max nor min).
        """

        mid_cycle_index = round(self.env.shape[0] // 4)
        tmp = self.env[mid_cycle_index, :]

        out_ = np.zeros(self.num_partials)

        # Interpolate for fractional bin values.
        for k in range(self.num_partials):
            frequency = (k + 1) * self.f0

            out_[k] = self.get_amp_from_frequency(frequency, tmp)

        return out_

    def _resample(self, tmp_env, frame_rate):
        return resample(tmp_env, frame_rate, self.sr)

    def synthesize(self):
        num_samples = self.get_num_samples()
        x = np.zeros(num_samples)

        if self.synth_mode == 'default' or self.synth_mode == 'raf':
            x = self.standard_synthesis(x)
        elif self.synth_mode == 'pam':
            x = self.pam_synthesis(x)
        else:
            raise ValueError("Unknown mode: {}.".format(self.synth_mode))
        return x

    def standard_synthesis(self, x):
        for k in np.arange(self.num_partials):
            x += self.make_partial(k)
        return x

    def pam_synthesis(self, x):
        """
        Returns a stimulus with a static spectral envelope, but having a global
        amplitude envelope of an equivalent spectrum-modulated signal.
        """

        average_gains = np.mean(self.processed_env, axis=0)

        # Sum all partial amplitudes into one master envelope.
        amp_envelope = np.sum(self.processed_env, axis=1)

        # Apply master envelope to each partial, scaling partials to average.
        for k in np.arange(self.num_partials):
            frequency = (k + 1) * self.f0
            gain = average_gains[k]

            x += gain * amp_envelope * self.make_carrier(frequency)
        return x

    def make_partial(self, k):
        frequency = (k + 1) * self.f0

        amp_envelope = self.processed_env[:, k]
        carrier = self.make_carrier(frequency)

        return amp_envelope * carrier

    def get_bin_num(self, frequency):
        return frequency / (self.sr // 2) * self.env.shape[1]

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

        # Randomize initial phase.
        phi = 2 * np.pi * np.random.rand()

        phase = np.cumsum(2 * np.pi * trajectory / self.sr) + phi
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
        Modulation depth from 0 to 1 based on `mod_hold` and `mod_fade` times.
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
    def __init__(
            self,
            env: np.ndarray,
            pr: int = PITCH_RATE,
            sr: int = SAMPLE_RATE,
            f0: float = None
    ):
        assert env.ndim == 2
        self.env = copy(env)

        assert pr > 0
        self.pr = pr

        assert sr > 0
        self.sr = sr

        if f0 is not None:
            assert 0 < f0 <= (sr // 2)
            self.f0 = f0
        else:
            self.f0 = None

        # Log tracks randomization settings, and order of morphing.
        self._log = []

    def time_average(self):
        num_frames, num_bins = self.env.shape

        tmp = np.mean(self.env, axis=0)
        tmp = np.tile(tmp, [num_frames, 1])

        self.env = copy(tmp)

    def shuffle_phase(self, num_shifts: int = 4):
        """
        Randomly shuffle each column.
        """

        assert num_shifts > 0

        all_shifts = np.linspace(0, 1, num_shifts, endpoint=False)
        num_frames, num_bins = self.env.shape

        # Used for pairing bins that surround a partial of interest.
        bins_above_partials = None
        if self.f0 is not None:
            bins_above_partials = self.get_bins_above_partials()

        tmp_log = []
        last_shift = None

        for k in np.arange(num_bins):

            # If necessary, match this shift to the previous bin.
            if self.f0 and (k in bins_above_partials):
                shift = last_shift
            else:
                shift = np.random.choice(all_shifts)

            tmp_log.append(
                {
                    'bin': k,
                    'shift': shift
                }
            )

            tmp = self.env[:, k]
            tmp = self.roll(tmp, shift)

            self.env[:, k] = tmp

            last_shift = copy(shift)

        self._log.append(tmp_log)

    def get_bin_num(self, frequency):
        return frequency / (self.sr // 2) * self.env.shape[1]

    def get_bins_above_partials(self):
        """Collect bins around overtone partials (higher in frequency only).

        In shuffling conditions, this allows for some bins to be shuffled in
        sync with one another, to avoid artifacts in synthesis later on.
        """

        out_ = []

        max_partial = int(
            (self.sr // 2) // self.f0
        )

        for p in range(1, max_partial + 1):
            frequency = self.f0 * p
            bin_number = self.get_bin_num(frequency)

            if bin_number % 1 != 0:
                out_.append(math.ceil(bin_number))

        return out_

    def get_bin_frequency(self, k):
        num_bins = self.env.shape[1]
        return k / num_bins * (self.sr / 2)

    def rap(self, max_random_gain: float = None):
        """Single cycle at base-rate with (possibly) randomized gains."""

        num_frames, num_bins = self.env.shape

        # Find average gain values in the cycle for each bin.
        ave_envelope = np.mean(self.env, axis=0)

        tmp_log = []

        for bin_ in range(num_bins):

            if max_random_gain:
                # Pick random gain (in dB) between 0 and `max_gain`.
                mod_gain = np.random.rand() * max_random_gain
            else:
                # Approximate partial-wise gains from envelope.
                mod_gain = self.get_partial_gain(bin_)

            # Build modulator.
            modulator = np.cos(
                2 * np.pi * np.linspace(0, 1, num_frames, endpoint=False)
            )

            modulator *= self.db_to_linear_coefficient(mod_gain)
            modulator += 1.

            # Multiply by base envelope gain.
            modulator *= ave_envelope[bin_]

            tmp_log.append(
                {
                    'bin': bin_,
                    'mod_gain': mod_gain
                }
            )

            # Place in array.
            self.env[:, bin_] = modulator

        self._log.append(tmp_log)

    def show(self, zoom=None):
        tmp = self.env

        if zoom:
            num_bins = tmp.shape[1]
            tmp = tmp[:, :(num_bins // zoom)]

        plot_envelope(tmp, show=True)

    def get_partial_gain(self, bin_):
        """Calculate the modulation gain depth, by partial, in decibels."""
        max_ = np.max(self.env[:, bin_], axis=0)
        min_ = np.min(self.env[:, bin_], axis=0)
        return 20 * np.log10(max_/min_ + EPS)

    def __str__(self):
        """Print the morph log."""
        s = f"""Summary\n-------\n\nTotal morphs:\t{len(self._log)}\n"""
        for morph in self._log:
            for bin_ in morph:
                s += f"\n{bin_}"
        return s

    def __call__(self):
        return self.env

    @staticmethod
    def db_to_linear_coefficient(decibels):
        a = 10. ** (decibels / 20) - 1
        b = 10. ** (decibels / 20) + 1
        return a / b

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
        FM depth in semitones, calculated as half the difference of pitch.
        """
        max_ = np.max(_datum['f0'])
        min_ = np.min(_datum['f0'])
        return 12 * np.log2(max_/min_) / 2

    # Synthesis parameters.
    partials = 70

    # Midi 48 -> C3.
    midi_pitch = 48

    synth_out = []

    for datum in single_cycles:
        fm_depth = get_fm_depth(datum)
        f0_ = midi_to_hz(midi_pitch)

        # Bring to linear amplitude. Env is calculated as the power spectrum.
        env_ = np.sqrt(datum['env'])

        morpher = EnvelopeMorpher(env_, f0=f0_)
        morpher.rap(max_random_gain=10)
        morpher.shuffle_phase(num_shifts=4)

        generator = StimulusGenerator(sr=SAMPLE_RATE, pr=PITCH_RATE)
        x = generator(
            f0=f0_,
            fm_depth=0.0,
            env=morpher(),
            num_partials=partials,
            length=2.1,
            mod_rate=5.,
            mod_hold=0.3,
            mod_fade=0.7,
            synth_mode='default',
            audio_fade=0.25,
        )

        stft_plot(x)

        synth_out.append(
            {
                'filename': datum['filename'],
                'f0': f0_,
                'wav': x,
            }
        )
