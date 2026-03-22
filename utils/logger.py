"""
Structured logging with rich formatting and file output.

Provides a unified logger that writes to both console (with color) and
a rotating file handler.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


_LOG_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)-25s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Create or retrieve a named logger with console and optional file output.

    Parameters
    ----------
    name : str
        The logger name (typically ``__name__``).
    level : int
        Logging level (default: ``logging.INFO``).
    log_dir : str or None
        Directory for log files.  If *None* no file handler is attached.
    max_bytes : int
        Maximum size per log file before rotation.
    backup_count : int
        Number of rotated log files to keep.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler with colour-coded output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler with rotation
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / f"{name}.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
