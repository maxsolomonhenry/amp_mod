"""
Tools for extracting timbral features from the stimuli.
"""

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
import os
import pandas as pd
from tqdm import tqdm

from src.defaults import DATA_PATH, TIMBRE_TOOLBOX_PATH, SYN_PATH
from src.util import matlab2np, np2matlab, read_wav, save_pickle


def extract_trials(df):
    df = df[df['trial_type'] == 'audio-slider-response']
    df = df[~df['stimulus'].str.contains("train")]
    return df


def init_matlab():
    print('Starting Matlab engine...')
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(TIMBRE_TOOLBOX_PATH))
    return eng


def load(datapath=DATA_PATH):
    pattern = os.path.join(datapath, 'prolific/*.csv')
    files = glob(pattern)
    assert files, 'No csv data found.'

    df = pd.DataFrame()

    for file in files:
        df = df.append(pd.read_csv(file))

    return df


def replace_path_to_local(_path):
    """
    Replace path with path to local file.
    """
    dir_, file_ = os.path.split(_path)

    tmp = dir_.split("/")
    tmp[0] = SYN_PATH
    tmp = '/'.join(tmp)

    return os.path.join(tmp, file_)


def timbre_toolbox(filepath, _eng):
    """
    Coupled to script `pyTimbre.m`
    """

    _data = _eng.pyTimbre(filepath, nargout=5)

    descriptor_names = [
        'energy',
        'spectral_centroid',
        'spectral_crest',
        'spectral_flatness',
        'odd_even_ratio',
    ]

    out_ = {}

    for i, _datum in enumerate(_data):
        out_[descriptor_names[i]] = matlab2np(_datum)

    return out_


if __name__ == '__main__':

    # Pickle file paths.
    timbretoolbox_name = 'TT_features.pickle'
    timbretoolbox_pickle_path = os.path.join(DATA_PATH, timbretoolbox_name)

    auditory_name = 'modulation_features.pickle'
    auditory_pickle_path = os.path.join(DATA_PATH, auditory_name)

    # Generate data.
    eng = init_matlab()

    df = load()
    df = extract_trials(df)

    paths = df['stimulus'].tolist()

    all_tt_data = []

    for path in tqdm(paths):
        localpath = replace_path_to_local(path)

        tt_data = timbre_toolbox(localpath, eng)
        tt_data['stimulus'] = path

        all_tt_data.append(tt_data)

    save_pickle(timbretoolbox_pickle_path, all_tt_data, force=False)

    # if debug:
    #     from src.util import load_pickle
    #     import pandas as pd
    #
    #     tmp = load_pickle(pickle_path)
    #     tmp = pd.DataFrame(tmp)
    #
    #     test = pd.merge(tmp, df, on='stimulus')
    #     print(test)
