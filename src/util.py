"""
General utilities.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import warnings

from scipy.io import wavfile
from scipy.signal import hilbert

from defaults import EPS


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


def load_data(path: str):
    assert os.path.isfile(path), 'Missing pickle. Run analysis.py'

    with open(path, 'rb') as handle:
        return pickle.load(handle)


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize array by max value.
    """
    return x / np.max(np.abs(x))


def read_wav(path: str):
    # Suppress wavfile complaints.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sample_rate, x = wavfile.read(path)
    return sample_rate, x


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
    smoothing: int = 1024
) -> np.ndarray:
    """
    Trims beginning of audio signal until it passes a given threshold in dB.
    """
    amplitude_envelope = np.abs(hilbert(signal))
    smoothed = np.convolve(amplitude_envelope, np.ones(smoothing)/smoothing)
    log_envelope = np.log(smoothed + EPS)
    start_index = np.maximum(
        np.where(log_envelope >= threshold)[0][0] - smoothing//2,
        0
    )
    return signal[start_index:]
