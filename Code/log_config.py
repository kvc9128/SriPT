import logging
from colorlog import ColoredFormatter

def setup_logging(log_file='SriPT.log'):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Log everything, regardless of level

    # File handler - logs all levels, append not overwrite, no color
    fh = logging.FileHandler(log_file, mode='a')  # 'a' for append mode
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    # Console handler - colored logging for warnings and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # Log warnings and above to console

    # Colored formatter for console
    console_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        log_colors={
            'INFO': 'green',
            'WARNING': 'blue',
            'ERROR': 'yellow',
            'CRITICAL': 'red',
        })
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)


# Set up the logging
setup_logging()
