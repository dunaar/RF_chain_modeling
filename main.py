#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: rf-chain-modeling
Description: Entry point of the project. Executes the rf_chain_example.
Author: Pessel Arnaud
Date: 2026-05-02
Version: 0.1.2.dev1
License: MIT
"""

import logging
from examples import rf_chain_example

logger = logging.getLogger(__name__)

def main() -> None:
    """Executes the primary RF chain example.

    This function sets up the root logger to output informational messages
    and triggers the demonstration script provided in the examples directory.

    Args:
        None

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s-%(levelname)s-%(module)s-%(funcName)s: %(message)s'
    )
    
    logger.info("Starting the RF chain modeling example execution.")
    
    try:
        rf_chain_example.main()
        logger.info("RF chain modeling example execution finished successfully.")
    except Exception as execution_error:
        logger.error("An error occurred during execution: %s", execution_error)

if __name__ == "__main__":
    main()