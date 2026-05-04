from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


#as seen in the example
SAMPLE_RATE = 48000


def extract_features(wav_path):
    """Load one wav file and return a dict of its features."""

    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE)
    features = {}

   