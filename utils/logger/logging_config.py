# utils/logger/logging_config.py

import os
import logging
from colorama import Fore, Style, Back, init

# Initialize colorama
init()

class ColorfulFormatter(logging.Formatter):
    # Define color mapping for each log level
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT
    }

    def format(self, record):
        # Apply color based on the log level
        log_color = self.COLORS.get(record.levelno, Fore.WHITE)
        record_msg = self._format(record)
        return log_color + record_msg + Style.RESET_ALL

    def _format(self, record):
        # Use the default formatter to create the log message string
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s')
        return formatter.format(record)    

def setup_logging(log_level):
    # ensure log directory exists
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # convert string log level to logging level
    log_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create a logger
    logger = logging.getLogger('Logger')
    logger.setLevel(log_level)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create and set up the file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s'))

    # Create and set up the console handler with color formatting
    console_handler = logging.StreamHandler()

    # Set the fomatter to the console handler
    console_handler.setFormatter(ColorfulFormatter('%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s'))

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Initialize and expose the logger
logger = setup_logging('DEBUG')