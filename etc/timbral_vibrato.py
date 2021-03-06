# -*- coding: utf-8 -*-
# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

from IPython.display import Audio


# +
# Helpers

def add_fade(
    signal: np.ndarray,
    fade_length: float,
    rate: int
):
    """
    Adds raised cosine fade in/out to signal.
    """
    num_samples = int(fade_length * rate)

    # Build ramp.
    t = np.linspace(0, 0.5, num_samples, endpoint=True)
    ramp = 0.5 * np.cos(2 * np.pi * t) + 0.5

    mean = np.mean(signal)
    signal -= mean

    # Fade in/out.
    signal[:num_samples] *= ramp[::-1]
    signal[-num_samples:] *= ramp

    signal += mean

    return signal

def MPS(x, win_size=2048, hop=256, eps=1e-8):
    """
    Convenient function to quickly have a peek at the modulation power spectrum.
    """
    X = stft(x, nperseg=win_size, noverlap=win_size-hop, nfft=win_size*2)[2]
    mX = np.abs(X)
    return np.fft.fft2(mX)


# -

class TimbralVibrato:
    """
    Oscillator with amplitude-modulated partials, simulating vibrato.
    """
    def __init__(
        self, 
        f0, 
        f_mod, 
        num_harmonics, 
        log_depth=-10,
        db_per_octave=-6, 
        random_frequency_minmax=(3, 10),
        dur=2, 
        sr=44100
    ):
        assert 30 <= f0 <= 5000
        assert 1 <= f_mod <= 20
        assert num_harmonics * f0 <= (sr / 2)
        assert log_depth >= 0
        assert 0 < dur

        self.f0 = f0
        self.f_mod = f_mod
        self.num_harmonics = num_harmonics
        self.log_depth = log_depth
        self.db_per_octave = db_per_octave
        self.dur = dur
        self.sr = sr
        self.random_frequency_minmax = random_frequency_minmax

        self.log = []

    def __call__(
        self, 
        fade=False, 
        wait=False,
        random_fm=False, 
        modulation_probability=0.5,
        fix_phase=False,
        randomize_gains=True,
        non_mod_amp=0.5,
        random_modulator=False
    ):
        self.log = []
        t = np.arange(0, self.dur, 1/self.sr)
        x = np.zeros(t.shape)

        amp_phase = [0, np.pi/2, np.pi, -np.pi/2]
        # amp_phase = [0, np.pi]

        alpha = self.alpha(self.db_per_octave)

        for k in range(1, self.num_harmonics + 1):

            # Random switches for amplitude modulation initial phase.
            phase_choice = np.random.randint(0, len(amp_phase))
            ap = amp_phase[phase_choice]

            # ap = np.random.rand() * np.pi * 2

            if fix_phase:
                ap = 0

            # Randomly apply amplitude modulation.
            mod_on = self.one_or_zero(modulation_probability)

            # Switch to randomize modulation frequency.
            f_mod = None

            if random_fm:
                tmp_min = self.random_frequency_minmax[0]
                tmp_span = self.random_frequency_minmax[1] - tmp_min
                f_mod = np.random.rand() * tmp_span + tmp_min
            else:
                f_mod = self.f_mod

            tmp_partial = np.cos(2 * np.pi * (t * k * self.f0 + np.random.rand()))

            if mod_on:
                if random_modulator:
                    tmp_amplitude = self.get_random_modulator(f_mod, t)
                else:
                    tmp_amplitude = np.cos(2 * np.pi * t * f_mod + ap)
            else:
                tmp_amplitude = non_mod_amp

            if fade and mod_on:
                wait_samps = int(wait * self.sr)
                fade_samps = int(fade * self.sr)
                tmp_amplitude *= np.concatenate(
                    [
                     np.zeros(wait_samps),
                     np.linspace(0, 1, fade_samps),
                     np.ones(len(t) - fade_samps - wait_samps)
                    ]
                )

            depth = None
            if mod_on:
                tmp_amplitude, depth = self.apply_depth(tmp_amplitude, randomize_gains)
            else:
                depth = 0

            x += tmp_partial * tmp_amplitude / (k ** alpha)
            self.log.append(
                {
                    'partial': k,
                    'is_modulating': mod_on,
                    'mod_phase': ap / np.pi,
                    'depth': np.around(depth, 4)
                }
            )
        return x

    def get_random_modulator(self, f_mod, t, num_harmonics=3):
        tmp = t * 0;
        for k in range(1, num_harmonics + 1):
            rnd_phase = np.random.rand() * 2 * np.pi
            rnd_gain = np.random.rand()
            tmp += rnd_gain * np.cos(2*np.pi*k*f_mod*t + rnd_phase) / k ** 2
        
        tmp = tmp / np.max(np.abs(tmp))
        return tmp
    
    def apply_depth(self, signal, randomize=True):
        depth = 1
        if randomize:
            depth = np.random.rand()
        tmp_depth = depth * self.log_depth
        A = (10 ** (tmp_depth / 20) - 1) / 2
        return (A * (1 + signal) + 1, depth)

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def one_or_zero(p_one):
        return np.abs(np.ceil(np.random.rand() - (1 - p_one)))

    @staticmethod
    def alpha(db_per_octave):
        """
        Calculate a coefficient such that `(1/k)**alpha` leads to a
        `db_per_octave` spectral slope, where `k` is the partial number.
        """
        return - db_per_octave / (20 * np.log10(2))


# +
my_osc = TimbralVibrato(
    f0=116.5, 
    f_mod=5,
    log_depth=10, 
    db_per_octave=-9,
    dur=2.3, 
    num_harmonics=60
)

x = my_osc(
    wait=0.3,
    fade=0.7, 
    random_fm=False, 
    modulation_probability=1, 
    fix_phase=False,
    randomize_gains=True,
    non_mod_amp=0.5,
    random_modulator=False
)
y = add_fade(x, 0.1, rate=my_osc.sr)

display(Audio(y, rate=my_osc.sr))
# -


