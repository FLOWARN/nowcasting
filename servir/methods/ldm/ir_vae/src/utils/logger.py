import logging
import sys
from datetime import datetime
import os

def setup_logging(output_dir: str) -> None:
    """
    Set up logging configuration with both file and console output.
    
    Args:
        output_dir: Directory where log files should be saved
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"training_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Logging setup complete. Log file: {log_file}")
        
    except Exception as e:
        print(f"Failed to setup logging: {str(e)}")
        raise