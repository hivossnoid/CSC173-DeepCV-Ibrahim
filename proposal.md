## CSC173 Deep Computer Vision ProjectCSC173 Deep Computer Vision Project 

- ProposalStudent: Abdallah Ibrahim, 2022-1684 
- Date: 12/11/2025

## Project Title

- Glossa Vision: Real-Time Pose-Based Sign Language Recognition

## Problem Statement

- Automated, real-time translation of sign language into natural language text remains a critical technical hurdle, essential for fostering communication and accessibility. Current camera-based systems often fail to generalize due to variations in signer identity, background complexity, and viewpoint changes. This project aims to simplify the problem by focusing on isolated American Sign Language (ASL) words, using robust pose estimation techniques to extract key kinematic features (hand and body movement) that are less susceptible to visual noise than raw pixel data. The goal is to develop a highly accurate, low-latency model to provide a viable real-time communication aid.

## Objectives

- Develop and train a deep computer vision model achieving a Top-1 Sign Classification Accuracy of at least 85% on a set of 50 isolated signs. 

- Implement a complete training pipeline including data preprocessing, pose keypoint extraction (using MediaPipe or a similar tool), data normalization, model training, validation, and evaluation. 

- Optimize the model for low-latency real-time inference (target $\le 500ms$ delay) to demonstrate practical usability.

## Dataset Plan

- Source: WLASL-100/300 (Word-Level American Sign Language) Classes: 100 Target Glosses (A diverse selection of 100 high-frequency ASL words, phrases, and the complete finger-spelling alphabet). Expected Size: ~4,000 to 5,000 video instances (Leveraging the full WLASL-300 subset or the WLASL-1000 with filtering). Acquisition: Download the WLASL dataset from the public repository (e.g., Kaggle/official GitHub). The dataset provides pre-segmented video clips ready for keypoint processing.

## Technical Approach

- Architecture SketchInput: Raw video frames (RGB). 

- Feature Extraction: MediaPipe Holistics (or Google's MoveNet for body) is applied to each frame to detect and extract $\approx 468$ facial, 33 body, and $2 \times 21$ hand keypoints. 

- Data Preprocessing: Normalize the 2D/3D keypoint coordinates relative to a fixed body reference (e.g., torso center) to ensure scale and positional invariance.

- Sequence Model: The normalized keypoints (features) across time are fed into a Temporal Graph Convolutional Network (TGCN) or a Transformer Encoder Block designed for sequence classification.Output: A one-hot encoded vector representing the predicted sign/word.Model: TGCN (Temporal Graph Convolutional Network) or LSTM-based Classifier (Pre-trained on WLASL-100/300, then fine-tuned on the combined dataset).

- Framework: PyTorch (for the sequence model) and Python/OpenCV (for real-time pipeline/keypoint extraction). Hardware: Lenovo LOQ

## Expected Challenges & Mitigations
- Challenge: Small dataset
- Solution: Augmentation
