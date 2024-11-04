import torch
import pandas as pd
import csv
import os
import numpy as np
from torch.utils.data import DataLoader
import logging

from data.dataset import COVID19Dataset
from data.preprocess import select_feat, normalize_data
from models.model import My_Model
from options.options import config
from options.util import same_seed, train_valid_split
from utils.logger import setup_logger

def load_test_data():
    """
    Load test data from CSV file.
    
    Returns:
        np.ndarray: Test data array
        
    Raises:
        TypeError: If test data is not a numpy array
    """
    test_data = pd.read_csv(os.path.join(config['data_dir'], config['test_file'])).values
    if not isinstance(test_data, np.ndarray):
        raise TypeError("Test data must be numpy array")
    return test_data

def predict(test_loader, model, device):
    """
    Make predictions using the trained model.
    
    Args:
        test_loader (DataLoader): DataLoader containing test data
        model (nn.Module): Trained PyTorch model
        device (torch.device): Device to run predictions on (CPU/GPU)
        
    Returns:
        np.ndarray: Array of predictions
        
    Raises:
        Exception: If any error occurs during prediction
    """
    try:
        model.eval()  # Set model to evaluation mode
        preds = []
        with torch.no_grad():  # Disable gradient calculation for inference
            for x in test_loader:
                x = x.to(device)
                pred = model(x)
                preds.append(pred.cpu())
        preds = torch.cat(preds, dim=0).numpy()
        return preds
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise e

def save_pred(preds, file):
    """
    Save predictions to a CSV file.
    
    Args:
        preds (np.ndarray): Array of predictions
        file (str): Path to output CSV file
    """
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])  # Write header
        for i, p in enumerate(preds):
            writer.writerow([i, p])  # Write each prediction with its index

def check_gpu_memory():
    """
    Check and print GPU memory usage if GPU is available.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache
        print(f'GPU memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB')

def main():
    """
    Main function for model testing and prediction.
    
    This function:
    1. Sets up logging and random seed
    2. Loads and preprocesses test data
    3. Loads trained model
    4. Makes predictions
    5. Saves results
    
    Raises:
        Exception: If any error occurs during the process
    """
    try:
        # Initialize logger
        logger = setup_logger('test.log')
        logger.info('Starting prediction process...')
        
        # Set random seed for reproducibility
        same_seed(config['seed'])
        
        # Set up device (GPU if available, else CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {device}')
        
        # Monitor GPU memory usage
        check_gpu_memory()

        # Load test data
        test_data = load_test_data()
        logger.info(f'Test data size: {test_data.shape}')

        # Load training data for feature selection reference
        train_data = pd.read_csv(os.path.join(config['data_dir'], config['train_file'])).values
        train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

        # Select features and normalize data
        x_train, x_valid, x_test, y_train, y_valid = select_feat(
            train_data, 
            valid_data, 
            test_data, 
            select_all=config['select_all']
        )
        x_train, mean, std = normalize_data(x_train)
        x_valid = (x_valid - mean) / (std + 1e-8)
        x_test = (x_test - mean) / (std + 1e-8)

        # Create test dataset and dataloader
        test_dataset = COVID19Dataset(x_test)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            pin_memory=True  # Enable faster data transfer to GPU
        )

        # Load trained model
        try:
            model = My_Model(input_dim=x_train.shape[1]).to(device)
            
            # Verify model file exists
            if not os.path.exists(config['save_path']):
                raise FileNotFoundError(f"Model file {config['save_path']} not found")
            
            # Load model checkpoint
            checkpoint = torch.load(config['save_path'], weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info('Model loaded successfully from {}'.format(config['save_path']))
            
            model.eval()  # Set model to evaluation mode
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e

        # Make predictions
        preds = predict(test_loader, model, device)
        
        # Save predictions to CSV
        output_path = os.path.join('predictions', 'pred.csv')
        os.makedirs('predictions', exist_ok=True)
        save_pred(preds, output_path)
        logger.info(f'Predictions saved to {output_path}')
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e

if __name__ == '__main__':
    main()