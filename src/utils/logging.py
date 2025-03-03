import logging
import os
from datetime import datetime

def setup_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with console and file handlers.

    Args:
        name: Name of the logger (e.g., module name).
        log_dir: Directory to save log files (default: 'logs').
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:  # Avoid duplicate handlers
        return logger

    # Formatter for log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (unique file per run)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# Example usage (can be imported in other files)
if __name__ == "__main__":
    logger = setup_logger("test_logger")
    logger.info("Logging setup complete.")
    logger.warning("This is a test warning.")
    logger.error("This is a test error.")