from pathlib import Path
import pandas as pd


modality_values = {
    "01": "full_AV",
    "02": "video-only",
    "03": "audio-only",
}
vocal_channel_values = {
    "01": "speech",
    "02": "song",
}
emotion_values = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
intensity_values = {
    "01": "normal",
    "02": "strong",
}
statement_values = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}
repetition_values = {
    "01": "1st",
    "02": "2nd",
}

def build_metadata(data_dir="data", output_path="data/processed/metadata.csv"):

    rows = []
    wav_files = sorted(data_dir.rglob("*.wav"))

    if not wav_files:
        print(f"WARNING: No .wav files found in {data_dir.resolve()}")
        print("Make sure you've run download_data.py first.")

    for file in wav_files:
        parts = file.stem.split("-")

        if len(parts) != 7:
            print("Invalid file")
            continue

        actor = int(parts[6])
        gender = "male" if actor % 2 == 1 else "female"

        rows.append({
            "filename":            file.name,
            "filepath":            str(file),
            "modality_number":     parts[0],
            "modality":            modality_values.get(parts[0]),
            "vocal_channel_number": parts[1],
            "vocal_channel":       vocal_channel_values.get(parts[1]),
            "emotion_code":        parts[2],
            "emotion":             emotion_values.get(parts[2]),
            "intensity_number":    parts[3],
            "intensity":           intensity_values.get(parts[3]),
            "statement_number":    parts[4],
            "statement":           statement_values.get(parts[4]),
            "repetition_number":   parts[5], 
            "repetition":          repetition_values.get(parts[5]),
            "actor":               actor,
            "gender":              gender,
        })

    metadata_table = pd.DataFrame(rows)
    metadata_table.to_csv(output_path, index=False)

    print(f"Saved metadata to: {output_path}")
    print(f"Shape: {metadata_table.shape}")
    print(metadata_table.head())

    return metadata_table


if __name__ == "__main__":
    build_metadata()