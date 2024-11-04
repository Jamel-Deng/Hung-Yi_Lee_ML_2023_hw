import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import logging
from utils.logger import setup_logger

# Import from other project modules
from data.dataset import COVID19Dataset
from data.preprocess import select_feat, normalize_data
from models.model import My_Model
from options.options import config
from options.util import same_seed, train_valid_split

def save_checkpoint(epoch, model, optimizer, loss, path):
    """
    Save model checkpoint to disk.
    
    Args:
        epoch (int): Current epoch number
        model (nn.Module): PyTorch model to save
        optimizer (torch.optim.Optimizer): Optimizer state to save
        loss (float): Current loss value
        path (str): Path to save checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    """
    Load model checkpoint from disk.
    
    Args:
        model (nn.Module): PyTorch model to load weights into
        optimizer (torch.optim.Optimizer): Optimizer to load state into
        path (str): Path to checkpoint file
        
    Returns:
        tuple: (epoch, loss) from checkpoint
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def save_training_progress(epoch, model, optimizer, scheduler, loss, best_loss, path):
    """
    Save complete training state including scheduler.
    
    Args:
        epoch (int): Current epoch number
        model (nn.Module): Current model state
        optimizer (torch.optim.Optimizer): Current optimizer state
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        loss (float): Current loss value
        best_loss (float): Best loss achieved so far
        path (str): Path to save progress file
    """
    progress = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'best_loss': best_loss
    }
    torch.save(progress, path)

def load_training_progress(model, optimizer, scheduler, path):
    """
    Load complete training state including scheduler.
    
    Args:
        model (nn.Module): Model to load state into
        optimizer (torch.optim.Optimizer): Optimizer to load state into
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler to load state into
        path (str): Path to progress file
        
    Returns:
        tuple: (epoch, loss, best_loss) from saved progress
    """
    progress = torch.load(path)
    model.load_state_dict(progress['model_state_dict'])
    optimizer.load_state_dict(progress['optimizer_state_dict'])
    scheduler.load_state_dict(progress['scheduler_state_dict'])
    return progress['epoch'], progress['loss'], progress['best_loss']

def trainer(train_loader, valid_loader, model, config, device):
    """
    Main training loop for the model.
    
    Args:
        train_loader (DataLoader): DataLoader for training data
        valid_loader (DataLoader): DataLoader for validation data
        model (nn.Module): Model to train
        config (dict): Training configuration parameters
        device (torch.device): Device to train on (CPU/GPU)
    """
    logger = logging.getLogger(__name__)
    try:
        # Initialize loss function and optimizer
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(model.parameters(), 
                                   lr=config['learning_rate'], 
                                   momentum=0.7)
        
        # Initialize learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',           # Reduce LR when monitored quantity stops decreasing
            factor=0.1,           # Multiply LR by this factor
            patience=5,           # Number of epochs with no improvement after which LR will be reduced
            verbose=True          # Print message for each update
        )
        
        # Initialize TensorBoard writer
        writer = SummaryWriter()

        # Create models directory if it doesn't exist
        if not os.path.isdir('./models'):
            os.makedirs('./models', exist_ok=True)

        # Initialize training variables
        n_epochs = config['n_epochs']
        best_loss = math.inf
        step = 0
        early_stop_count = 0
        progress_path = os.path.join('./models', 'progress.pt')

        # Resume training if progress file exists
        if os.path.exists(progress_path):
            start_epoch, current_loss, best_loss = load_training_progress(
                model, optimizer, scheduler, progress_path)
            logger.info(f'Resuming training from epoch {start_epoch}')
            start_epoch += 1
        else:
            start_epoch = 0
            logger.info('Starting training from scratch')

        # Main training loop
        for epoch in range(start_epoch, n_epochs):
            # Training phase
            model.train()
            loss_record = []
            train_pbar = tqdm(train_loader, position=0, leave=True)

            for x, y in train_pbar:
                # Forward pass and loss calculation
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Record loss and update progress bar
                step += 1
                loss_record.append(loss.detach().item())
                train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
                train_pbar.set_postfix({'loss': loss.detach().item()})

            # Calculate average training loss
            mean_train_loss = sum(loss_record)/len(loss_record)
            writer.add_scalar('Loss/train', mean_train_loss, step)

            # Validation phase
            model.eval()
            loss_record = []
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    pred = model(x)
                    loss = criterion(pred, y)
                loss_record.append(loss.item())
                
            # Calculate average validation loss
            mean_valid_loss = sum(loss_record)/len(loss_record)
            logger.info(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
            writer.add_scalar('Loss/valid', mean_valid_loss, step)

            # Save best model and update early stopping
            if mean_valid_loss < best_loss:
                best_loss = mean_valid_loss
                save_checkpoint(epoch, model, optimizer, best_loss, config['save_path'])
                logger.info(f'Saving model with loss {best_loss:.3f}...')
                early_stop_count = 0
            else: 
                early_stop_count += 1

            # Save training progress
            save_training_progress(epoch, model, optimizer, scheduler, 
                                 mean_valid_loss, best_loss, progress_path)

            # Early stopping check
            if early_stop_count >= config['early_stop']:
                logger.info('\nModel is not improving, so we halt the training session.')
                return

            # Update learning rate
            scheduler.step(mean_valid_loss)
            
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise e

def check_data_dir():
    """
    Verify that data directory and required files exist.
    
    Raises:
        FileNotFoundError: If data directory or required files are not found
    """
    if not os.path.exists(config['data_dir']):
        raise FileNotFoundError(f"Data directory {config['data_dir']} not found")
    
    train_path = os.path.join(config['data_dir'], config['train_file'])
    test_path = os.path.join(config['data_dir'], config['test_file'])
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file {train_path} not found")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file {test_path} not found")

def print_config():
    """
    Print training configuration parameters.
    """
    logger = logging.getLogger(__name__)
    logger.info("\nTraining Configuration:")
    logger.info("-" * 20)
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("-" * 20 + "\n")

def check_gpu_memory():
    """
    Check and log GPU memory usage if GPU is available.
    """
    logger = logging.getLogger(__name__)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f'GPU memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB')

def main():
    """
    Main function to run the training pipeline.
    
    This function:
    1. Sets up logging and environment
    2. Loads and preprocesses data
    3. Initializes model and training components
    4. Runs training loop
    
    Raises:
        Exception: If any error occurs during the process
    """
    try:
        # Setup logging and initial checks
        logger = setup_logger()
        logger.info('Starting training process...')
        check_data_dir()
        check_gpu_memory()
        print_config()
        
        # Set random seed for reproducibility
        same_seed(config['seed'])
        
        # Set device (GPU/CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        
        # Load and preprocess data
        logger.info('Loading data...')
        train_data = pd.read_csv(os.path.join(config['data_dir'], config['train_file'])).values
        test_data = pd.read_csv(os.path.join(config['data_dir'], config['test_file'])).values
        
        # Split training data
        train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])
        
        # Log dataset sizes
        logger.info(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")
        
        # Feature selection and normalization
        x_train, x_valid, x_test, y_train, y_valid = select_feat(
            train_data, 
            valid_data, 
            test_data, 
            config['select_all']
        )
        x_train, mean, std = normalize_data(x_train)
        x_valid = (x_valid - mean) / (std + 1e-8)
        x_test = (x_test - mean) / (std + 1e-8)
        
        logger.info(f'Number of features: {x_train.shape[1]}')
        
        # Create datasets and dataloaders
        train_dataset = COVID19Dataset(x_train, y_train)
        valid_dataset = COVID19Dataset(x_valid, y_valid)
        test_dataset = COVID19Dataset(x_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,          # Shuffle training data
            pin_memory=True        # Speed up data transfer to GPU
        )
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,         # Don't shuffle validation data
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,
            pin_memory=True
        )
        
        # Initialize model and optimizer
        model = My_Model(input_dim=x_train.shape[1]).to(device)
        optimizer = torch.optim.SGD(model.parameters(), 
                                  lr=config['learning_rate'], 
                                  momentum=0.7)
        
        # Load checkpoint if exists
        if os.path.exists(config['save_path']):
            start_epoch, best_loss = load_checkpoint(model, optimizer, config['save_path'])
            logger.info(f'Resuming training from epoch {start_epoch}')
        else:
            logger.info('Starting training from scratch')
        
        logger.info(f'Model structure: {model}')
        
        # Start training
        logger.info('\nStarting training...')
        trainer(train_loader, valid_loader, model, config, device)
        logger.info('Training completed!')
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e

if __name__ == '__main__':
    main()