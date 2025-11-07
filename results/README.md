# Results Directory

This directory stores evaluation results, plots, and performance metrics.

## Contents

After running the notebook, this directory will contain:

### Performance Metrics
- `evaluation_metrics.json` - Comprehensive metrics (accuracy, precision, recall, F1, AUC)
- `predictions.csv` - Model predictions on test set
- `training_log.csv` - Training history per epoch

### Visualizations
- `confusion_matrix.png` - Confusion matrix heatmap
- `roc_curve.png` - ROC curve with AUC score
- `training_history.png` - Training/validation accuracy and loss curves
- `detection_report_*.json` - Real-time detection analysis

### Dataset-Specific Results
```
results/
├── RWF-2000/
│   ├── evaluation_metrics.json
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── Hockey/
├── Surveillance/
└── inference/
    └── detection_reports/
```

## Note

Result files (images, CSVs, JSONs) are excluded from Git. 
Only the directory structure is maintained.
