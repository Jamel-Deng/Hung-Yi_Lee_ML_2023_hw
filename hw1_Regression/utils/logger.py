import logging

def setup_logger(log_file='train.log'):
    """
    Configure and set up a logger for both file and console output.
    
    This function initializes a logger that writes messages to both a file
    and the console with timestamp, log level, and message information.
    
    Args:
        log_file (str): Path to the log file. Defaults to 'train.log'
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        >>> logger = setup_logger('my_training.log')
        >>> logger.info('Training started')
        2024-03-21 10:30:15 - INFO - Training started
    """
    # Configure the basic settings for logging
    logging.basicConfig(
        # Set the logging level to INFO to capture all info, warning, and error messages
        level=logging.INFO,
        
        # Define log message format:
        # %(asctime)s: Timestamp
        # %(levelname)s: Level of the log message (INFO, WARNING, ERROR, etc.)
        # %(message)s: The actual log message
        format='%(asctime)s - %(levelname)s - %(message)s',
        
        # Set up multiple handlers to output logs to both file and console
        handlers=[
            logging.FileHandler(log_file),     # Write logs to file
            logging.StreamHandler()            # Write logs to console/stdout
        ]
    )
    
    # Return a logger instance with the name of the current module
    return logging.getLogger(__name__)