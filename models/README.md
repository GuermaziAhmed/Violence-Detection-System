# Models Directory

This directory stores trained model files.

## Contents

After training, this directory will contain:
- `best_model.h5` - Best performing model checkpoint
- `training_info.json` - Training metadata and hyperparameters
- `c3d_model.h5` - Full C3D model
- `lightweight_model.h5` - Optimized lightweight variant
- `*.tflite` - TensorFlow Lite models for mobile/edge deployment
- `*.onnx` - ONNX format for cross-platform deployment

## Note

Model files are excluded from Git (see `.gitignore`) due to their large size (15MB-100MB each).

Download pre-trained models from: [Release Page Link]
