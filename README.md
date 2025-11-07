# Violence Detection System

Deep learning project for detecting violence in videos.

## What It Does

Detects violent behavior in surveillance videos using 3D-CNN neural networks.

## Features

- Works with multiple datasets (RWF-2000, Hockey Fight, etc.)
- Multiple model options (C3D, ConvLSTM, BiConvLSTM)
- Real-time video processing
- Automated alerts when violence is detected  

## Installation

```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-learn pandas numpy
```

## Quick Start

Open `violence_detection_system.ipynb` and run the cells.



## Datasets

Download datasets and put them in the `data/` folder:
- RWF-2000: https://www.kaggle.com/datasets/vulamnguyen/rwf2000
- Hockey Fight: https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes

## Project Structure

```
violence_detection_system.ipynb  # Main notebook
data/                            # Video datasets
models/                          # Trained models
results/                         # Evaluation results
logs/                            # Training logs
```

