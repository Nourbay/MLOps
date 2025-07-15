import logging
import logging.handlers
import os
from datetime import datetime

def setup_logging(log_file='mlflow.log', level=logging.INFO):
    """
    Setup logging configuration for the application.
    This creates a log file that can be consumed by Logstash.
    
    Args:
        log_file (str): Path to the log file
        level: Logging level (default: INFO)
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except PermissionError:
            # Fallback to current directory if permission denied
            log_file = 'mlflow.log'
            print(f"Warning: Could not create logs directory. Using {log_file} instead.")
    
    # Configure logging with error handling
    try:
        logging.basicConfig(
            level=level,
            format='%(asctime)s %(levelname)s %(name)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                # File handler for Logstash consumption
                logging.FileHandler(log_file, mode='a', encoding='utf-8'),
                # Console handler for immediate feedback
                logging.StreamHandler()
            ]
        )
    except PermissionError:
        # Fallback to console-only logging if file creation fails
        print(f"Warning: Could not create log file {log_file}. Using console logging only.")
        logging.basicConfig(
            level=level,
            format='%(asctime)s %(levelname)s %(name)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler()
            ]
        )
    
    # Create a logger for this application
    logger = logging.getLogger('mlflow_app')
    logger.setLevel(level)
    
    return logger

def get_logger(name='mlflow_app'):
    """
    Get a logger instance.
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name) 