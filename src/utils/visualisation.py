# Create new file: src/utils/logger.py

import logging
import os

def setup_logger(output_dir: str) -> logging.Logger:
    """Setup logger for training - MISSING FUNCTION"""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, 'training.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger