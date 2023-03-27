"""
logger.py

This module provides a logger that can be used to write log messages to a file.

Usage:
------
To use the logger, import the `logger` object from this module and call its methods
to write log messages.
"""

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

handler = logging.FileHandler('greenwood.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)