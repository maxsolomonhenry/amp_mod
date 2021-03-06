import numpy as np
import os
from scipy.io import wavfile
from tqdm import tqdm

from src import macro
from src.analysis import single_cycles
from src.defaults import SAMPLE_RATE, SYN_PATH
from src.util import midi_to_hz, safe_mkdir


# Helper.
def quick_write(_file_path, _filename, _data):
    """Write file as 16bit mono PCM."""

    write_path = os.path.join(_file_path, _filename)

    amplitude = np.iinfo(np.int16).max
    _data *= amplitude

    wavfile.write(write_path, SAMPLE_RATE, _data.astype(np.int16))


# Experiment parameters.
num_subjects = 200
num_blocks = 2
repeats_per_block = 4

# Use this to start counting from a subject number greater than 0.
starting_subject = 200

# Load env as linear amplitude. (CheapTrick calculates the power spectrum.)
env = single_cycles[0]['env']
env = np.sqrt(env)

# Synthesis parameters.
synthesis_params = {
    'num_partials': 70,
    'f0': midi_to_hz(48),
    'fm_depth': 0.1314,
    'length': 2.5,
    'mod_rate': 5.,
    'mod_hold': 0.,
    'mod_fade': 0.,
    'audio_fade': 0.25,
    'env': env,
}

for s in range(num_subjects):
    s += starting_subject
    print(f"\nGenerating stimuli for subject {s}...")

    # Make subject directory.
    subject_path = os.path.join(SYN_PATH, f"subject_{s}/")
    safe_mkdir(subject_path)

    # Open log.
    log_path = os.path.join(subject_path, f"stimlog_subject_{s}.txt")
    log = open(log_path, "w")

    log.write(f"Subject: {s}\n" + "-" * 10 + "\n")

    for b in tqdm(range(num_blocks)):

        # Make block directory.
        block_path = os.path.join(subject_path, f"block_{b}/")
        safe_mkdir(block_path)

        log.write("\n" + "="*7 + f"\nBlock {b}\n" + "="*7 + "\n")

        for r in range(repeats_per_block):
            # Generate one of each kind of stimulus.

            # BASIC.
            tmp_x = macro.make_basic(synthesis_params)
            quick_write(block_path, f"BASIC_{r}.wav", tmp_x)

            # FROZEN.
            tmp_x = macro.make_frozen(synthesis_params)
            quick_write(block_path, f"FROZEN_{r}.wav", tmp_x)

            # FM-ONLY.
            tmp_x = macro.make_fm_only(synthesis_params)
            quick_write(block_path, f"FM_ONLY_{r}.wav", tmp_x)

            # SHUFFLE and SHUFFLE RAF.
            tmp_x, tmp_x_raf = macro.make_shuffle(synthesis_params, log)
            quick_write(block_path, f"SHUFFLE_{r}.wav", tmp_x)
            quick_write(block_path, f"SHUFFLE_RAF_{r}.wav", tmp_x_raf)

            # SIMPLE and SIMPLE RAF.
            tmp_x, tmp_x_raf = macro.make_simple(synthesis_params, log)
            quick_write(block_path, f"SIMPLE_{r}.wav", tmp_x)
            quick_write(block_path, f"SIMPLE_RAF_{r}.wav", tmp_x_raf)

            # RAG and RAG RAF.
            tmp_x, tmp_x_raf = macro.make_rag(synthesis_params, log)
            quick_write(block_path, f"RAG_{r}.wav", tmp_x)
            quick_write(block_path, f"RAG_RAF_{r}.wav", tmp_x_raf)

            # PAM.
            tmp_x = macro.make_pam(synthesis_params)
            quick_write(block_path, f"PAM_{r}.wav", tmp_x)

            # Control.
            tmp_x = macro.make_control(synthesis_params)
            quick_write(block_path, f"CONTROL_{r}.wav", tmp_x)

    log.close()
