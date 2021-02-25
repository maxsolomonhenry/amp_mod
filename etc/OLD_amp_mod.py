# +
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd

from vowel_spectra import spectra, data


# +
# TODO: fade in modulation (McAdams 1989 found this aids in the effect).

class AmpModStimulus:
    def __init__(
        self,
        dur:float = 2., 
        fm: float = 5.5,
        mod_depth: float = 10.,
        sr:int = 44100
    ):        
        assert dur > 0.4
        self.dur = dur
        
        assert fm > 0.
        self.fm = fm
        
        assert mod_depth >= 0.
        self.mod_depth = mod_depth

        assert sr > 0
        self.sr = sr
        
        self.t = self.init_t()
    
    def init_t(self):
        return np.arange(0, self.dur, 1/self.sr)
    
    def make_sine(self, f, random_phase=True):
        phi = random_phase * 2 * np.pi  * np.random.rand()
        return np.cos(2 * np.pi * f * self.t + phi)
    
    def apply_modulation(self, signal, rnd_fm, rnd_phase, fade):
        # Apply modulation as additive `mod_depth` in dB's.
        tmp_modulator = self.generate_modulator(rnd_fm, rnd_phase)
        tmp_amp = self.get_amplitude_coefficient(self.mod_depth)
        amp = self.fade_to(tmp_amp, fade)
        return (amp * (1 + tmp_modulator) + 1) * signal
    
    def fade_to(self, value, length):
        tmp_ramp = np.linspace(0, 1, np.int(length * self.sr), endpoint=False)
        tmp_sustain = np.ones(len(self.t) - len(tmp_ramp))
        return np.concatenate([tmp_ramp, tmp_sustain]) * value
    
    def generate_modulator(self, rnd_fm, rnd_phase):
        fm = None
        if rnd_fm:
            # Generate random modulation frequency between 3 and 15 Hz.
            fm = np.random.rand() * 12. + 3.
        else:
            fm = self.fm

        phi = None
        if rnd_phase:
            phi = 2 * np.pi * np.random.rand()
        else:
            phi = 0
            
        return np.cos(2 * np.pi * self.t * fm + phi)
    
    def get_formant_gain(self, frequency, vowel):
        spectrum = spectra[vowel]
        bin_num = frequency / self.sr * len(spectrum)
        
        tmp_fraction = bin_num % 1
        tmp_a = spectrum[int(np.floor(bin_num))]
        tmp_b = spectrum[int(np.ceil(bin_num))]
        
        return (1 - tmp_fraction) * tmp_a + tmp_fraction * tmp_b
    
    def __call__(self, f0, vowel, rnd_fm=False, rnd_phase=False, num_harmonics=50, fade=0.4):
        assert num_harmonics * f0 <= (self.sr / 2)
        assert fade < self.dur
        
        x = np.zeros(self.t.shape)
        
        for k in range(1, num_harmonics + 1):
            tmp_partial = self.make_sine(k*f0)
            tmp_partial *= self.get_formant_gain(k * f0, vowel)
            tmp_partial = self.apply_modulation(tmp_partial, rnd_fm, rnd_phase, fade)
            x += tmp_partial
        
        return x
    
    @staticmethod
    def get_amplitude_coefficient(depth):
        return (10 ** (depth / 20) - 1) / 2

def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


# +
def normalize(x):
    return x / np.max(np.abs(x))

rnd_fm = False
rnd_phase = True

stim1 = AmpModStimulus(dur=4, fm=3.1, mod_depth=12)
x1 = stim1(midi_to_hz(48), rnd_fm=rnd_fm, rnd_phase=rnd_phase, vowel="a", num_harmonics=80)

stim2 = AmpModStimulus(dur=4, fm=4.01, mod_depth=15)
x2 = stim2(midi_to_hz(53), rnd_fm=rnd_fm, rnd_phase=rnd_phase, vowel="e", num_harmonics=80)

stim3 = AmpModStimulus(dur=4, fm=7, mod_depth=12)
x3 = stim3(midi_to_hz(58), rnd_fm=rnd_fm, rnd_phase=rnd_phase, vowel="i", num_harmonics=80)

all_ = normalize(x1) + normalize(x2) + normalize(x3)

ipd.Audio(all_, rate=stim1.sr)
# +
# Skecthpad.
import resampy

spectra['VC3']

def get_formant_gain(frequency, spectrum, audio_sr=44100):
    bin_num = frequency / audio_sr * len(spectrum)

    tmp_fraction = bin_num % 1
    tmp_a = spectrum[int(np.floor(bin_num))]
    tmp_b = spectrum[int(np.ceil(bin_num))]

    return (1 - tmp_fraction) * tmp_a + tmp_fraction * tmp_b

def harmonic_quantize(f0_trajectory, spectrum, fundamental=110, audio_sr=44100, num_harmonics=50):
    y = np.zeros(f0_trajectory.shape)
    for k in range(1, num_harmonics + 1):
        q_f0 = np.tile(k * fundamental, f0_trajectory.shape)
        
        amps = np.zeros(f0_trajectory.shape)
        for i, f0 in enumerate(f0_trajectory):
            # Generate amplitude envelope.
            amps[i] = get_formant_gain(f0, spectrum)

        phase = np.cumsum(2*np.pi*q_f0/audio_sr)
        y += np.cos(phase) * amps
    
    return y
        


# +
spec = data[0]['world']['sp'][20,:] # spectra['VC3']

spec = spec / np.max(np.abs(spec))

f0_track = (1 + 0.1 * np.cos(2 * np.pi * 5 * np.arange(0, 2, 1/44100))) * 110

x = harmonic_quantize(f0_track, spec, num_harmonics=50)
# -

ipd.Audio(x, rate=44100)


