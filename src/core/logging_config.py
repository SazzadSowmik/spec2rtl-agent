"""
Logging configuration for Spec2RTL-Agent.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Setup logging configuration.
    
    Creates two log files:
    - full_log_{timestamp}.log: Everything (DEBUG level)
    - summary_{timestamp}.log: Important milestones (INFO level)
    
    Args:
        log_dir: Directory to store log files
        
    Returns:
        Logger instance
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logger
    logger = logging.getLogger("spec2rtl")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Format
    detailed_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_format = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Handler 1: Full debug log to file
    full_log_file = log_path / f"full_log_{timestamp}.log"
    full_handler = logging.FileHandler(full_log_file, encoding='utf-8')
    full_handler.setLevel(logging.DEBUG)
    full_handler.setFormatter(detailed_format)
    logger.addHandler(full_handler)
    
    # Handler 2: Summary log to file (INFO and above)
    summary_log_file = log_path / f"summary_{timestamp}.log"
    summary_handler = logging.FileHandler(summary_log_file, encoding='utf-8')
    summary_handler.setLevel(logging.INFO)
    summary_handler.setFormatter(simple_format)
    logger.addHandler(summary_handler)
    
    # Handler 3: Console output (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_format)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized")
    logger.info(f"Full log: {full_log_file}")
    logger.info(f"Summary log: {summary_log_file}")
    
    return logger