from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

#We are using librosa because it is what breaks down each wav file
#into our sample rate and makes frames in which we can extract features from

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

    #so the loop feature basically because its a 2d matrix we get the summary stats for each row 1-13,
    #That is why we have the mfcc[i], so we get 4 summary stats per iteration = 4 * (loop number)
    #it then puts that info into the features field. For those that dont require loops we get only 4.
    #this whole thing works by putting in a way file one by one and extracting all features.

    #MFCC 13 coefficents from the wav file. Which goes and shows exactly the speakers mouth position and air flow
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = mfcc[i].mean()
        features[f"mfcc_{i+1}_std"]  = mfcc[i].std()
        features[f"mfcc_{i+1}_min"]  = mfcc[i].min()
        features[f"mfcc_{i+1}_max"]  = mfcc[i].max()
    
    #interesting this is the delta of the original MFCC 
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    for i in range(13):
        features[f"mfcc_delta_{i+1}_mean"] = mfcc_delta[i].mean()
        features[f"mfcc_delta_{i+1}_std"]  = mfcc_delta[i].std()
        features[f"mfcc_delta_{i+1}_min"]  = mfcc_delta[i].min()
        features[f"mfcc_delta_{i+1}_max"]  = mfcc_delta[i].max()

    #this is the delta of the delta of the MFCC
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(13):
        features[f"mfcc_delta2_{i+1}_mean"] = mfcc_delta2[i].mean()
        features[f"mfcc_delta2_{i+1}_std"]  = mfcc_delta2[i].std()
        features[f"mfcc_delta2_{i+1}_min"]  = mfcc_delta2[i].min()
        features[f"mfcc_delta2_{i+1}_max"]  = mfcc_delta2[i].max()

    #From my understanding this will give us the pitch
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        features[f"chroma_{i+1}_mean"] = chroma[i].mean()
        features[f"chroma_{i+1}_std"]  = chroma[i].std()
        features[f"chroma_{i+1}_min"]  = chroma[i].min()
        features[f"chroma_{i+1}_max"]  = chroma[i].max()

    #From my understanding this is a step to MFCC but kept as its own feature. We convert to DB to have 
    #a log scale so loud pieces dont dominate the summary stats.
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    for i in range(64):
        features[f"mel_{i+1}_mean"] = mel_db[i].mean()
        features[f"mel_{i+1}_std"]  = mel_db[i].std()
        features[f"mel_{i+1}_min"]  = mel_db[i].min()
        features[f"mel_{i+1}_max"]  = mel_db[i].max()

    #the amount of times the waveform crosses 0
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = zcr.mean()
    features["zcr_std"]  = zcr.std()
    features["zcr_min"]  = zcr.min()
    features["zcr_max"]  = zcr.max()

    #Room mean square energy
    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = rms.mean()
    features["rms_std"]  = rms.std()
    features["rms_min"]  = rms.min()
    features["rms_max"]  = rms.max()

    #Center of mass of a spectrum in hz
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spec_centroid_mean"] = centroid.mean()
    features["spec_centroid_std"]  = centroid.std()
    features["spec_centroid_min"]  = centroid.min()
    features["spec_centroid_max"]  = centroid.max()
    
    #The spread between the centroid
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spec_bandwidth_mean"] = bandwidth.mean()
    features["spec_bandwidth_std"]  = bandwidth.std()
    features["spec_bandwidth_min"]  = bandwidth.min()
    features["spec_bandwidth_max"]  = bandwidth.max()
   
    #Where 85% of the energy lies below.
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    features["spec_rolloff_mean"] = rolloff.mean()
    features["spec_rolloff_std"]  = rolloff.std()
    features["spec_rolloff_min"]  = rolloff.min()
    features["spec_rolloff_max"]  = rolloff.max()

    return features

def build_feature_table(metadata_csv="data/processed/metadata.csv",output_csv="data/processed/features.csv"):
    

    metadata = pd.read_csv(metadata_csv)
    print(f"Loaded metadata with {len(metadata)} rows")

    rows = []
    skipped = 0
    #didnt like that you couldnt see what was going on, found TQDM and looks much nicer.
    for _, meta_row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting features"):

        feats = extract_features(meta_row["filepath"])

        # If the file was corrupt, extract_features returned None — skip it
        if feats is None:
            skipped += 1
            continue

        # Combine the metadata fields with the new feature fields
        row = meta_row.to_dict()
        row.update(feats)
        rows.append(row)

    feature_table = pd.DataFrame(rows)

    feature_table = feature_table.dropna()

    feature_table.to_csv(output_csv, index=False)

    print(f"\nSaved features to: {output_csv}")
    print(f"Shape: {feature_table.shape}")

    return feature_table

if __name__ == "__main__":
    build_feature_table()



