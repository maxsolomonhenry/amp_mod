import numpy as np
import os
from scipy.io import wavfile

from src.analysis import single_cycles
from src.defaults import PITCH_RATE, SAMPLE_RATE, SYN_PATH
from src.synthesis import EnvelopeMorpher, StimulusGenerator
from src.util import midi_to_hz, safe_mkdir

# Load env as linear amplitude. (CheapTrick calculates the power spectrum.)
env = single_cycles[0]['env']
env = np.sqrt(env)

# TODO parseargs

# Experiment parameters.
num_subjects = 1
num_blocks = 4
repeats_per_block = 2

# Synthesis parameters.
num_partials = 70
midi_pitch = 48
fm_depth = 0.1314
length = 2.
mod_rate = 5.
mod_hold = 0.3
mod_fade = 0.7

# Calculations.
f0 = midi_to_hz(midi_pitch)

# TODO create log as writing.

for s in range(num_subjects):

    # Make subject directory.
    file_path = os.path.join(SYN_PATH, f"subject_{s}/")
    safe_mkdir(file_path)

    for b in range(num_blocks):

        # Make block directory.
        file_path = os.path.join(file_path, f"block_{b}/")
        safe_mkdir(file_path)

        for r in range(repeats_per_block):
            # Generate one of each kind of stimulus.

            filename = f"BASIC_{r}.wav"
            write_path = os.path.join(file_path, filename)
            wavfile.write(write_path, SAMPLE_RATE, data)