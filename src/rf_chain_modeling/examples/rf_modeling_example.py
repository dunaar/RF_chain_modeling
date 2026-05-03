#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project:     rf-chain-modeling
Module:      examples.rf_modeling_example
Description: Demonstration of the RF modeling core classes and complex signal generation.
Author:      Pessel Arnaud
Date:        2026-05-02
License:     MIT
Usage:       python3 -m examples.rf_modeling_example
"""

import logging
from math import pi

import matplotlib.pyplot as plt
import numpy as np

from rf_chain_modeling.rf_utils.rf_modeling import Signals, thermal_noise_power_dbm, calculate_spectral_rms_dbm

logger = logging.getLogger(__name__)


def main() -> None:
    """Main function to demonstrate the usage of the RF modeling classes."""
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s : %(message)s"
    )

    # Create a signal with noise and tones
    signal = Signals(20e9, 100e6)  # fmax = 20 GHz, bin_width = 100 MHz (duration = 10 ns)

    # Add thermal noise
    signal.add_noise(thermal_noise_power_dbm(signal.temp_kelvin, signal.bandwidth))

    # Add tones
    tones = [(17e9, -10, -pi / 3), (11e9, -5, pi / 4), (3e9, 0, 0)]
    for fr, pow, ph in tones:
        logger.info("Adding %4.1f GHz tone at %4.1f dBm", fr / 1e9, pow)
        signal.add_tone(fr, pow, ph)
        logger.info("signal.rms_dbm() after adding %4.1f GHz tone: %6.2f dBm", fr / 1e9, signal.rms_dbm())

    # Print RMS value
    logger.info("%6.2f dBm in whole spectrum",
                calculate_spectral_rms_dbm(signal.spectrums.mean(axis=0),
                                           signal.imped_ohms))

    for fr, pow, ph in tones:
        idx_neg = signal.get_arg_freq(-fr)
        idx_pos = signal.get_arg_freq( fr)
        logger.info("%6.2f dBm in spectrum at %4.1f GHz",
                    calculate_spectral_rms_dbm(signal.spectrums[:, (idx_neg, idx_pos)].mean(axis=0),
                                               signal.imped_ohms), fr / 1e9)

    # Plot temporal signal
    #signal.plot_temporal(tmax=10e-9)

    # Plot spectrum
    #signal.plot_spectrum()

    # Display all plots
    plt.show()


if __name__ == "__main__":
    main()