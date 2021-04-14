from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer

from defaults import DATA_PATH, MAX_INTEGER


def anova_prep(df):
    df = df.groupby(['subjectNo', 'condition'])['response'].mean()
    df = df.unstack()
    tmp = []
    for subject in df.index:
        for condition in df.loc[subject].keys():
            tmp.append(
                {
                    'subjectNo': subject,
                    'condition': condition,
                    'rating': df.loc[subject, condition],
                }
            )
    return pd.DataFrame(tmp)


def average_condition_rating_within_subject(df):
    tmp = df.groupby(['subjectNo', 'condition'])['response'].mean()
    return tmp.unstack()


def average_std_of_ratings(df):
    return df.groupby(['subjectNo', 'condition'])['response'].std().\
        groupby('condition').mean()


def box_plot(df, study_type, savefig=False, dpi=300):
    tmp = df.groupby(['subjectNo', 'condition'])['response'].mean()
    tmp = tmp.unstack()
    tmp.columns = [s.replace('_', ' ') for s in tmp.columns]
    plt.figure(figsize=(16, 6))
    sns.boxplot(data=tmp)
    plt.ylabel('rating')
    plt.xlabel('condition')
    if savefig:
        plt.savefig(f"figs/study_type_{study_type}_boxplot.png", dpi=dpi)
    else:
        plt.show()


def extract_condition(df):
    if 'condition' not in df:
        def process(x):
            return "_".join(x[-1].split('_')[:-1])

        df['condition'] = df['stimulus'].str.split('/').apply(process)
    return df


def extract_subject(df):
    if 'subjectNo' not in df:
        df['subjectNo'] = df['stimulus'].str.split("/").apply(
            lambda x: x[1].split("_")[1])
    return df


def extract_trials(df):
    df = df[df['trial_type'] == 'audio-slider-response']
    df = df[~df['stimulus'].str.contains("train")]
    return df


def filter_by_basic(df, threshold=0.6):
    """Find subjectNo where the BASIC condition was rated below threshold."""
    tmp1 = df[df['condition'] == 'BASIC'].groupby(['subjectNo'])[
               'response'].min() > threshold
    tmp2 = tmp1[tmp1]
    print(f"N = {len(tmp2)}")
    return df[df['subjectNo'].isin(tmp2.keys())]


def filter_by_control(df, threshold=0.6):
    """Find subjectNo where the CONTROL was rated greater than threshold."""
    tmp1 = df[df['condition'] == 'CONTROL'].groupby(['subjectNo'])[
               'response'].min() > threshold
    tmp2 = tmp1[tmp1]
    print(f"N = {len(tmp2)}")
    return df[df['subjectNo'].isin(tmp2.keys())]


def get_good_participants(num_reject=2):
    """Filter participants by number of rejections"""

    pattern = os.path.join(DATA_PATH, 'participant_demographic_data/*.csv')
    files = glob(pattern)

    df = pd.DataFrame()

    for i, file in enumerate(files):
        tmp = pd.read_csv(file)
        tmp['phase'] = i
        df = df.append(tmp)

    return df.query(f'status == "APPROVED" and num_rejections <= {num_reject}')[
        'participant_id']


def get_num_subjects(df):
    return len(df['subjectNo'].unique())


def get_summary(df):
    tmp1 = df[['condition', 'response']].groupby('condition').mean()
    tmp2 = df[['condition', 'response']].groupby('condition').std()

    tmp3 = pd.DataFrame()
    tmp3['mean'] = tmp1['response']
    tmp3['std'] = tmp2['response']
    return tmp3


def group_quantile_transform(series):
    quantiler = QuantileTransformer()
    return np.squeeze(quantiler.fit_transform(series.values.reshape(-1, 1)))


def isolate_study(df, study_type):
    assert not df[df['studyType'] == study_type].empty, 'Returns no trials.'
    return df[df['studyType'] == study_type]


def load(pattern='prolific/*.csv'):
    pattern = os.path.join(DATA_PATH, pattern)

    files = glob(pattern)
    assert files, 'No csv data found.'

    df = pd.DataFrame()

    for file in files:
        df = df.append(pd.read_csv(file))

    return df


def load_and_clean_data(num_reject=10000):
    df = load()
    df = extract_trials(df)
    df = normalize_slider(df)
    df = extract_subject(df)
    df = extract_condition(df)
    df = min_max_norm(df)

    # Filter by participants having few rejections.
    gp = get_good_participants(num_reject=num_reject)
    df = df[df['prolificID'].isin(gp)]

    df = df.reset_index(drop=True)
    return df.drop(
        ['view_history', 'trial_type', 'internal_node_id', 'studyID',
         'sessionID', 'url', 'slider_start'],
        1
    )


def load_tt_descriptors():
    tmp = pd.read_csv('./timbre_toolbox_features/Median_PowSTFTrep.csv')
    tmp.columns = ["STFT__" + col + "Med" for col in tmp.columns]
    stft_median_df = tmp.rename(columns={'STFT__SoundFileMed': 'stimulus'})

    tmp = pd.read_csv('./timbre_toolbox_features/IQR_HARMrep.csv')
    tmp.columns = ["HARMONIC__" + col + "IQR" for col in tmp.columns]
    harmonic_iqr_df = tmp.rename(columns={'HARMONIC__SoundFileIQR': 'stimulus'})

    tmp = pd.read_csv('./timbre_toolbox_features/IQR_PowSTFTrep.csv')
    tmp.columns = ["STFT__" + col + "IQR" for col in tmp.columns]
    stft_iqr_df = tmp.rename(columns={'STFT__SoundFileIQR': 'stimulus'})

    tmp = pd.read_csv('./timbre_toolbox_features/Median_HARMrep.csv')
    tmp.columns = ["HARMONIC__" + col + "Med" for col in tmp.columns]
    harmonic_med_df = tmp.rename(columns={'HARMONIC__SoundFileMed': 'stimulus'})

    tmp = pd.merge(stft_median_df, harmonic_iqr_df, on='stimulus')
    tmp = pd.merge(tmp, stft_iqr_df, on='stimulus')
    tmp = pd.merge(tmp, harmonic_med_df, on='stimulus')

    def reformat_stimulus(x):
        _tmp = x.replace('__', '/')
        return 'audio/' + _tmp + '.wav'

    tmp['stimulus'] = tmp['stimulus'].transform(reformat_stimulus)
    return tmp


def max_time_elapsed(df):
    """Returns the max time elapsed in minutes."""
    return df.groupby('subjectNo')['time_elapsed'].max() / 1000 / 60


def min_max_norm(df):
    min_ = df.groupby('subjectNo')['response'].transform('min')
    max_ = df.groupby('subjectNo')['response'].transform('max')
    df['response'] = (df['response'] - min_) / (max_ - min_)
    return df


def normalize_slider(df):
    already_normalized = (df[['response', 'slider_start']] <= 1).all().all()
    if not already_normalized:
        df[['response', 'slider_start']] = df[['response',
                                               'slider_start']] / MAX_INTEGER
    return df


def response_histograms(df, bins=20):
    # Get subject's average rating per condition.
    tmp = average_condition_rating_within_subject(df)
    for i, col in enumerate(tmp):
        plt.subplot(1, 2, (i % 2) + 1)
        plt.title(col)
        plt.hist(tmp[col], bins=bins)
        if i % 2 == 1:
            plt.show()


def within_subject_correlation(_df, _feature, _method):
    """Group by subject, get correlation with response, mean over subjects."""
    return _df.groupby('subjectNo')[_feature].corr(
        _df['response'], method=_method
    ).mean()
