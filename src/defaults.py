"""
Defaults and globals.
"""

import os


class RealPath:
    """
    Convenient way to generate absolute file-paths.
    """
    def __init__(self):
        self.here = os.path.dirname(__file__)

    def __call__(self, relative_path):
        return os.path.realpath(
            os.path.join(self.here, relative_path)
        )


# Small value.
EPS = 1e-8

# Sample rates.
SAMPLE_RATE = 44100
PITCH_RATE = 200

# Relevant file paths.
real_path = RealPath()

ANA_PATH = real_path('../audio/ana')
SYN_PATH = real_path('../audio/syn')
DATA_PATH = real_path('../data')
