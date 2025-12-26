# Project Title: Glossa Vision: Real-Time American Sign Language Recognition using Temporal Graph Convolutional Networks
**CSC173 Intelligent Systems Final Project** *Mindanao State University - Iligan Institute of Technology* **Student:** Abdallah Ibrahim, 2022-1684 
**Semester:** AY 2025-2026 Sem 1  
[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org)

## Abstract
This project addresses the communication gap between the Deaf community and the hearing public through a Real-Time Sign Language Recognition (SLR) system. By leveraging MediaPipe Holistic for robust 3D skeletal keypoint extraction and a Temporal Graph Convolutional Network (TGCN), the model interprets complex temporal dependencies in hand and body movements. The system was trained on the WLASL dataset subset, achieving low-latency inference suitable for practical deployment.

## Methodology
### Dataset
- Source: WLASL (World-Level American Sign Language) Dataset
- Classes: 1154 isolated signs
- Preprocessing: Scale-invariant shoulder-width normalization, 48-frame temporal windowing.

### Architecture
- **Feature Extractor:** MediaPipe Holistic (Pose, Face, Hands)
- **Model:** TGCN (Spatial Graph Convolution + Temporal LSTM/Conv)
- **Nodes:** 543 landmarks per frame

| Parameter | Value |
|-----------|-------|
| Sequence Length | 48 frames |
| Total Nodes | 543 |
| Batch Size | 32 (Adjusted for memory) |
| Optimizer | Adam |

## Experiments & Results
### Metrics
| Metric | Result |
|-------|---------|
| Total Classes | 1154 |
| Inference Latency (Model Only) | 11.95 ms |
| Total Pipeline Latency (End-to-End) | < 150 ms |

## Conclusion
The system successfully demonstrates the feasibility of GCNs for real-time sign language interpretation. While accuracy is high on specific classes, future work involves refining the dataset to balance class distributions and deploying the model.

## Demo
[./Glossa_Vision/demo/CSC173_Ibrahim_Final.mp4]

## Installation
1. `git clone <your-repo-link>`
2. `pip install torch mediapipe opencv-python numpy`
3. `python inference_tgcn.py`
## References
[1] Yan, S., et al. "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition," AAAI, 2018.  
[2] Li, D., et al. "Word-level Deep Sign Language Recognition from Video," WACV, 2020.

## Github
https://github.com/hivossnoid/CSC173-DeepCV-Ibrahim