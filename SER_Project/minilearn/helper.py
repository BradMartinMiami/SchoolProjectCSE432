import pandas as pd

def ezcols(): 

    features = pd.read_csv("data/processed/features.csv")

    metadata_cols = [
    "filename", "filepath", "modality", "modality_number",
    "vocal_channel", "vocal_channel_number",
    "emotion", "emotion_code",
    "intensity", "intensity_number",
    "statement", "statement_number",
    "repetition", "repetition_number",
    "actor", "gender"
    ]
    
    feature_actual = [col for col in features.columns if col not in metadata_cols]

    return feature_actual