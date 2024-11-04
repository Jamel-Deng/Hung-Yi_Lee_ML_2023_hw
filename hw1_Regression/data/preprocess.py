import numpy as np

def select_feat(train_data, valid_data, test_data, select_all=True):
    '''
    Selects useful features to perform regression.
    
    Args:
        train_data (np.ndarray): Training data with shape (n_samples, n_features + 1),
                                where the last column is the target variable
        valid_data (np.ndarray): Validation data with same format as train_data
        test_data (np.ndarray): Test data with shape (n_samples, n_features)
        select_all (bool): If True, select all features. If False, select only specific features.
                          Defaults to True.
    
    Returns:
        tuple: Contains:
            - x_train (np.ndarray): Selected features for training
            - x_valid (np.ndarray): Selected features for validation
            - x_test (np.ndarray): Selected features for testing
            - y_train (np.ndarray): Training targets
            - y_valid (np.ndarray): Validation targets
    '''
    # Extract target variables from train and validation data
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    # Extract feature matrices
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        # Use all available features
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        # Select specific feature columns starting from index 35
        feat_idx = list(range(35, raw_x_train.shape[1]))  # TODO: Select suitable feature columns.

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid

def normalize_data(data):
    """
    Normalizes the input data using z-score normalization (standardization).
    
    Args:
        data (np.ndarray): Input data to be normalized with shape (n_samples, n_features)
    
    Returns:
        tuple: Contains:
            - normalized_data (np.ndarray): The normalized data
            - mean (np.ndarray): Mean values used for normalization
            - std (np.ndarray): Standard deviation values used for normalization
    """
    # Calculate mean along each feature
    mean = np.mean(data, axis=0)
    # Calculate standard deviation along each feature
    std = np.std(data, axis=0)
    # Normalize data using z-score normalization
    # Add small epsilon (1e-8) to avoid division by zero
    normalized_data = (data - mean) / (std + 1e-8)
    
    return normalized_data, mean, std