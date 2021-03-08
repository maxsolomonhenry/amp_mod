"""
General utilities.
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import warnings

from librosa import load
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, hilbert
from typing import Union

from defaults import EPS, PITCH_RATE, SAMPLE_RATE


def add_fade(
    signal: np.ndarray,
    fade_length: float,
    rate: int,
    f_out: bool = False,
):
    """
    Adds linear fade in/out to signal.
    """

    num_samples = int(fade_length * rate)

    # Build ramp.
    ramp = np.linspace(0, 1, num_samples, endpoint=False)

    mean = np.mean(signal)
    signal -= mean

    # Fade in/out.

    if f_out:
        signal[-num_samples:] *= ramp[::-1]
    else:
        signal[:num_samples] *= ramp

    signal += mean

    return signal


def contains_nan(in_: np.ndarray) -> bool:
    return np.isnan(np.sum(in_))


def flatten(signal: np.ndarray) -> np.ndarray:
    """
    Replace an array values with its mean.
    """
    mean_ = np.mean(signal)
    return np.tile(mean_, len(signal))


def force_mono(signal: np.ndarray) -> np.ndarray:
    """
    Forces stereo signal to mono by averaging channels.
    """
    assert len(signal.shape) <= 2, "Mono or stereo arrays only, please."
    if len(signal.shape) == 2:
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T
        signal = np.mean(signal, axis=0)
    return signal


def get_amp_envelope(signal: np.ndarray, cutoff: float = 25., sr: int = 44100):
    amplitude_envelope = np.abs(hilbert(signal))
    smoothed = low_pass(amplitude_envelope, cutoff, sample_rate=sr, order=4)
    return smoothed


def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    """
    Converts from Hz to linear pitch space, where midi:69 = A440.
    """
    return np.maximum(0, 12 * np.log2((hz + EPS)/440) + 69)


def load_data(path: str):
    assert os.path.isfile(path), 'Missing pickle. Run analysis.py'

    with open(path, 'rb') as handle:
        return pickle.load(handle)


def low_pass(
        signal: np.ndarray,
        frequency: float,
        sample_rate: int,
        order: int = 16
):
    """
    Convenience function for butterworth lowpass filter.
    """
    Wn = frequency/(sample_rate / 2)
    [b, a] = butter(order, Wn, btype='lowpass')

    out_ = filtfilt(b, a, signal)
    assert not contains_nan(out_), "Filtering generated NaNs."

    return out_


def midi_to_hz(midi: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Converts from linear pitch space to Hz, where A440 = midi:69.
    """
    return 440.0 * (2.0**((midi - 69.0) / 12.0))


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize array by max value.
    """
    return x / np.max(np.abs(x))


def plot_envelope(env, show=True):
    plt.imshow(env.T, aspect='auto', origin='lower')
    if show:
        plt.show()


def read_wav(path: str):
    x, sample_rate = load(path, sr=SAMPLE_RATE, dtype=np.float64)
    return sample_rate, x


def resample(
        env: np.ndarray,
        frame_rate: float,
        sr: int,
) -> np.ndarray:
    """
    Resample spectral envelope array in time.
    """

    axis = -1
    if env.ndim == 2:
        axis = 0

    _indices = np.arange(env.shape[0]) * sr / frame_rate
    f = interp1d(_indices, env, kind='linear', axis=axis)

    num_samples = int(
        round((env.shape[0] - 1) * sr / frame_rate)
    )

    return f(np.arange(num_samples))


def save_data(path: str, data, force: bool = False):
    if force is False:
        assert not os.path.isfile(path), 'File {} already exists.'.format(
            os.path.basename(path)
        )
    else:
        warnings.warn('Forcing overwrite...')

    print('Saving file {}...'.format(os.path.basename(path)))
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def stft_plot(
    signal: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    title: str = "",
    show: bool = True
):
    X = librosa.stft(signal)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(5, 5))
    plt.title(title)
    librosa.display.specshow(Xdb, sr=sample_rate, x_axis="time", y_axis="linear")
    if show:
        plt.show()


def time_plot(
        signal: np.ndarray,
        rate: int = 44100,
        show: bool = True,
        title: str = None
):
    t = np.linspace(0, len(signal)/rate, len(signal), endpoint=False)
    plt.plot(t, signal)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    if title:
        plt.title(title)
    if show:
        plt.show()


def trim_to_duration(
    signal: np.ndarray,
    time_in: float = 1.,
    duration: float = 1.,
    rate: int = 44100
) -> np.ndarray:
    """
    Trim audio signal given start time and desired duration.
    """
    in_ = int(time_in * rate)
    out_ = in_ + int(duration * rate)
    return signal[in_:out_]


def trim_silence(
    signal: np.ndarray,
    threshold: float = -35,
    cutoff: float = 25.,
    sr: int = 44100,
) -> np.ndarray:
    """
    Trims beginning of audio signal until it passes a given threshold in dB.
    """
    amp_envelope = get_amp_envelope(signal, cutoff, sr)
    log_envelope = np.log(amp_envelope + EPS)
    start_index = np.maximum(
        np.where(log_envelope >= threshold)[0][0],
        0
    )
    return signal[start_index:]


def upsample(
        hz: np.ndarray,
        sr: int = SAMPLE_RATE,
        pr: int = PITCH_RATE
) -> np.ndarray:
    _indices = np.arange(len(hz)) * sr / pr
    f = interp1d(_indices, hz, kind='cubic')

    num_samples = int(
        round((len(hz) - 1) * SAMPLE_RATE / PITCH_RATE)
    )

    return f(np.arange(num_samples))
