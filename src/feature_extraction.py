"""
Tools for extracting timbral features from the stimuli.
"""

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

from src.defaults import SYN_PATH
from src.util import read_wav


def extract_trials(df):
    df = df[df['trial_type'] == 'audio-slider-response']
    df = df[~df['stimulus'].str.contains("train")]
    return df


def load(pattern='./prolific/*.csv'):
    files = glob(pattern)
    assert files, 'No csv data found.'

    df = pd.DataFrame()

    for file in files:
        df = df.append(pd.read_csv(file))

    return df


def replace_path_to_local(path):
    """Replace path with path to local file."""
    dir_, file_ = os.path.split(path)

    tmp = dir_.split("/")
    tmp[0] = SYN_PATH
    tmp = '/'.join(tmp)

    return os.path.join(tmp, file_)


if __name__ == '__main__':
    # Tests.

    df = load()
    df = extract_trials(df)

    paths = df['stimulus'].tolist()

    for path in paths:
        path = replace_path_to_local(path)
        sr, x = read_wav(path)

        print(sr)
