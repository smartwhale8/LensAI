import logging
import os

def setup_logging(log_level):
    # Ensure log directory exists
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Convert string log level to logging level
    log_level = getattr(logging, log_level.upper(), logging.INFO)

    # Define logging configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'app.log')),
            logging.StreamHandler()
        ]
    )