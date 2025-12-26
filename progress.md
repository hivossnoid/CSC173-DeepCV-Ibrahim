# CSC173 Deep Computer Vision ProjectCSC173 Deep Computer Vision Project 

- ProposalStudent: Abdallah Ibrahim, 2022-1684 
- Date: 12/11/2025

# Project Title

- Glossa Vision: Real-Time Pose-Based Sign Language Recognition


## 1. Dataset Progress

    - Source & Structure
        Data directory: ./pose_features
        Format: JSON files containing sequences of frames with 3D joint landmarks (x, y, z).
        Sequence length: 48 frames (padded or truncated as necessary).
        Each frame: flattened vector of all joints [V * 3].
    
    - Labels
        Automatically inferred from filename (prefix before underscore).
        Encoded using LabelEncoder.
        Number of unique classes: num_classes (from dataset).
        Default behavior: unknown labels mapped to "UNKNOWN".
        
    - Dataset Statistics
        Total JSON files loaded: len(files) (only valid non-empty JSONs).
        Total sequences: len(dataset) (after filtering empty/malformed files).
        Training / Validation split: 80% / 20%.
        Input features per timestep: [V * 3] (V = number of joints).
    
    - Preprocessing
        Centered joints on torso (mean of first 3 joint coordinates).
        Padded shorter sequences with the last frame.
        Converted to PyTorch tensors for DataLoader.
        
    - Observations
        Dataset is small enough to overfit a model easily (training accuracy ~92% while validation accuracy remains 0%).
        Likely class imbalance or distribution mismatch between training and validation sets.
        Many sequences may be repetitive due to padding, which can affect temporal modeling.

## 2. Training Progress

    | Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
    | ----- | ---------- | --------- | -------- | ------- |
    | 1     | 8.533      | 0%        | 8.273    | 0%      |
    | 10    | 7.924      | 0%        | 11.826   | 0%      |
    | 20    | 3.382      | 26.7%     | 14.653   | 0%      |
    | 30    | 0.763      | 83.8%     | 16.396   | 0%      |
    | 40    | 0.385      | 89.7%     | 17.649   | 0%      |
    | 50    | 0.238      | 92.3%     | 18.216   | 0%      |