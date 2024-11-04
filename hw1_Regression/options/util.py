import torch
import numpy as np
import random

def same_seed(seed):
    """
    Set random seed for reproducibility across different components.
    
    This function sets the same seed for PyTorch CPU operations, CUDA operations,
    NumPy operations, and Python's random module to ensure reproducible results.
    
    Args:
        seed (int): The random seed to be used
    """
    # Set PyTorch CPU seed
    torch.manual_seed(seed)
    
    # Set CUDA seed if GPU is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set Python's random module seed
    random.seed(seed)
    
    # Ensure deterministic behavior in CUDA operations
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_valid_split(data, valid_ratio, seed):
    """
    Split dataset into training and validation sets.
    
    This function performs a random split of the input data into training
    and validation sets based on the specified ratio.
    
    Args:
        data (np.ndarray): Input data to be split
        valid_ratio (float): Ratio of data to be used for validation (0.0 to 1.0)
        seed (int): Random seed for reproducible splitting
    
    Returns:
        tuple: Contains:
            - train_data (np.ndarray): Training dataset
            - valid_data (np.ndarray): Validation dataset
    """
    # Set random seed for reproducible splitting
    np.random.seed(seed)
    
    # Generate random permutation of indices
    indices = np.random.permutation(len(data))
    
    # Calculate size of validation set
    valid_size = int(len(data) * valid_ratio)
    
    # Split indices into validation and training sets
    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]
    
    # Return split datasets
    return data[train_indices], data[valid_indices] 