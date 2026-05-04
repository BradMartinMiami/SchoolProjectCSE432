from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


#as seen in the example
SAMPLE_RATE = 48000

#hopefully WAV_path lines up to CSV/dataframe
def extract_features(wav_path):
    """Load one wav file and return a dict of its features."""
    try:
        y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE)
    except Exception as e:
        print("Skipping bad file")
        return None
        
    #same thing like with rows when appending last time    
    features = {}

    #MFCC 13 coefficents from notes
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = mfcc[i].mean()
        features[f"mfcc_{i+1}_std"]  = mfcc[i].std()
        features[f"mfcc_{i+1}_min"]  = mfcc[i].min()
        features[f"mfcc_{i+1}_max"]  = mfcc[i].max()
    
    #MFCC delta
    mfcc_delta = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = mfcc[i].mean()
        features[f"mfcc_{i+1}_std"]  = mfcc[i].std()
        features[f"mfcc_{i+1}_min"]  = mfcc[i].min()
        features[f"mfcc_{i+1}_max"]  = mfcc[i].max()
   