import logging
from log_config import setup_logging  # This will configure the logging

# Get the logger
logger = logging.getLogger(__name__)

def some_function():
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")


# Example usage
some_function()
