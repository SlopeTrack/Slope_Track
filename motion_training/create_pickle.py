import os
import numpy as np
import pandas as pd
import pickle
import configparser  # <-- for reading seqinfo.ini

def read_seqinfo(seq_path):
    """
    Read imWidth and imHeight from seqinfo.ini
    """
    ini_path = os.path.join(seq_path, 'seqinfo.ini')
    config = configparser.ConfigParser()
    config.read(ini_path)

    width = int(config['Sequence']['imWidth'])
    height = int(config['Sequence']['imHeight'])
    fps = int(config['Sequence']['framerate']),
    return width, height, fps

# Function to check if values are between 0 and 1
def check_between_0_and_1(df, columns):
    for col in columns:
        if not ((df[col] >= 0) & (df[col] <= 1)).all():
            print(f"Column {col} has values outside [0, 1]")
        else:
            print(f"Column {col} is OK (all values between 0 and 1)")

def compute_time_difference(t: int, t_ref: int, fps: float) -> float:
    if fps <= 0:
        raise ValueError("FPS must be positive.")
    return (t - t_ref) / fps

def convert_mot_to_mamba_dataset(mot_folder, output_file, seq_len=5):
    """
    mot_folder: folder containing MOT-format subfolders (one folder per sequence)
    output_file: where to save processed Mamba training dataset (as pickle)
    seq_len: number of frames for input sequence (default: 5)
    """
    full_tracks = []

    # Loop through subfolders (sequences)
    for seq_folder in os.listdir(mot_folder):
        seq_path = os.path.join(mot_folder, seq_folder)
        gt_path = os.path.join(seq_path, "gt", "gt.txt")

        if not os.path.isfile(gt_path):
            continue

        print(f"Processing sequence: {seq_folder}...")

        # Read image size from seqinfo.ini
        try:
            im_width, im_height, fps = read_seqinfo(seq_path)
        except Exception as e:
            print(f"Warning: Could not read seqinfo.ini for {seq_folder}. Skipping.")
            continue

        df = pd.read_csv(gt_path, header=None)
        df.columns = ["frame", "track_id", "x", "y", "w", "h", "conf", "class", "vis"]

        # Convert xywh top-left to cxcywh
        df['cx'] = df['x'] + df['w'] / 2.0
        df['cy'] = df['y'] + df['h'] / 2.0

        """# Normalize bbox coordinates
        df['cx_norm'] = df['cx'] / im_width
        df['cy_norm'] = df['cy'] / im_height
        df['w_norm'] = df['w'] / im_width
        df['h_norm'] = df['h'] / im_height"""

        df['cx_norm'] = df['cx']
        df['cy_norm'] = df['cy']
        df['w_norm'] = df['w']
        df['h_norm'] = df['h']

        # List of columns to check
        #columns_to_check = ['cx_norm', 'cy_norm', 'w_norm', 'h_norm']

        # Perform the check
        #check_between_0_and_1(df, columns_to_check)

        # Reset track IDs to be sequential per sequence
        old_to_new_ids = {old_id: new_id for new_id, old_id in enumerate(sorted(df['track_id'].unique()))}
        df['new_track_id'] = df['track_id'].map(old_to_new_ids)


        # Group by new track ID
        track_ids = df['new_track_id'].unique()

        for tid in track_ids:
            track_data = df[df['new_track_id'] == tid].sort_values(by="frame")
            frames = []
            for _, row in track_data.iterrows():
                frame_id = int(row['frame'])
                bbox = [row['cx_norm'], row['cy_norm'], row['w_norm'], row['h_norm']]
                frames.append((frame_id, bbox))
            if len(frames) >= seq_len + 1:
                full_tracks.append({
                    'sequence': seq_folder,
                    'fps':fps,
                    'track_id': int(tid),
                    'old_track_id': int(track_data['track_id'].iloc[0]),
                    'frames': frames,
                    'width':im_width,
                    'height':im_height
                })

    

    print(f"Total tracklets collected: {len(full_tracks)}")

    # Save processed dataset as pickle
    with open(output_file, 'wb') as f:
        pickle.dump(full_tracks, f)
    print(f"Saved processed dataset to {output_file}")

if __name__ == "__main__":
    base_folder = "../slope_track"

    # Train
    train_folder = os.path.join(base_folder, "train")
    train_output = os.path.join("train_set.pkl")
    if os.path.isdir(train_folder):
        convert_mot_to_mamba_dataset(train_folder, train_output)
    else:
        print("⚠️ Train folder not found, skipping...")

    # Validation
    val_folder = os.path.join(base_folder, "val")
    val_output = os.path.join( "val_set.pkl")
    if os.path.isdir(val_folder):
        convert_mot_to_mamba_dataset(val_folder, val_output)
    else:
        print("⚠️ Validation folder not found, skipping...")
