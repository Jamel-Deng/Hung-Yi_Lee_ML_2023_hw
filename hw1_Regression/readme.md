# COVID-19 Case Prediction Model

A project for predicting COVID-19 case numbers based on deep learning.

## Directory Structure

'''
project-root/
├── data/                   # Data-related files
│   ├── raw/               # Raw data
│   │   ├── covid.train.csv
│   │   └── covid.test.csv
│   ├── dataset.py         # Dataset class definition
│   └── preprocess.py      # Data preprocessing functions
├── models/                # Model-related files
│   └── model.py           # Model definition
├── options/               # Configuration files
│   ├── options.py         # Configuration parameters
│   └── util.py            # Utility functions
├── utils/                 # Utility functions
│   └── logger.py          # Logging utility
├── train.py               # Training script
├── test.py                # Testing script
└── requirements.txt       # Project dependencies
'''

## Quick Start

### 1. Environment Setup
- `pip install -r requirements.txt`

### 2. Data Preparation

- Place the training data and test data in the `data/raw/` directory.

### 3. Train the Model

- `python train.py`  

### 4. Make Predictions 

- `python test.py` 

## Notes

1. Ensure CUDA is installed if using a GPU.
2. Training logs are saved in `train.log`
3. Prediction results are stored in `predictions/pred.csv`
4. The best model is saved as `models/model.ckpt`

