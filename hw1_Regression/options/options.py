config = {
    # Random seed for reproducibility
    'seed': 5201314,
    
    # Feature selection mode
    # If True: use all features
    # If False: use selected subset of features
    'select_all': True,
    
    # Ratio of training data to be used as validation set
    'valid_ratio': 0.2,
    
    # Training hyperparameters
    'n_epochs': 5000,        # Maximum number of training epochs
    'batch_size': 256,       # Number of samples per batch
    'learning_rate': 1e-5,   # Learning rate for optimizer
    'early_stop': 600,       # Number of epochs for early stopping patience
    
    # Model checkpoint saving path
    'save_path': './models/model.ckpt',
    
    # Data directory and file paths
    'data_dir': './data/raw',           # Directory containing the data files
    'train_file': 'covid.train.csv',    # Training data filename
    'test_file': 'covid.test.csv'       # Test data filename
} 