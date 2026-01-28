# Egocentric Data QC Using SAM3

Quality control pipeline for egocentric video data using SAM3 (Segment Anything Model 3) for hand detection and tracking.

## Overview

This project provides an end-to-end pipeline for quality control analysis of egocentric videos, specifically focused on detecting and tracking hands throughout video sequences. It uses Facebook Research's SAM3 model for video segmentation and provides metrics to evaluate video quality based on hand presence.

## Installation

### Prerequisites

- HuggingFace account with access to:
  - [Egocentric-10K dataset](https://huggingface.co/datasets/builddotai/Egocentric-10K)
  - [SAM3 model](https://huggingface.co/facebook/sam3)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd embodiment-qc
```

2. Install dependencies:
```bash
pip install torch torchvision
pip install opencv-python matplotlib scikit-learn
pip install 'git+https://github.com/facebookresearch/sam3.git'
pip install datasets huggingface_hub
```

3. Set up HuggingFace authentication:
```bash
export HF_TOKEN=your_huggingface_token_here
# Or use: huggingface-cli login
```

### Using with Egocentric-10K Dataset

See `notebooks/egocentric_qc_sam3_example.ipynb` for a complete example using the Egocentric-10K dataset.

## Acknowledgments

- [SAM3](https://github.com/facebookresearch/sam3) by Facebook Research
- [Egocentric-10K](https://huggingface.co/datasets/builddotai/Egocentric-10K) dataset
