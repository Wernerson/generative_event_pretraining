import os
root = "/data/storage/jianwen/DSEC/train_optical_flow"
for sequence in sorted(os.listdir(root)):
    sequence_path = os.path.join(root, sequence, "flow", "forward")
    timestamp = os.path.join(root, sequence, "flow", "forward_timestamps.txt")
    with open(timestamp, 'r') as f:
        lines = f.readlines()
    for i, frame in enumerate(sorted(os.listdir(sequence_path))):
        new_name = lines[i+1].split(",")[0] + ".png"
        print(f"Renaming {frame} to {new_name}")
        os.rename(os.path.join(sequence_path, frame), os.path.join(sequence_path, new_name))