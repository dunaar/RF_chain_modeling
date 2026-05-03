#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project:     rf-chain-modeling
Module:      tests.test_rf_modeling
Description: Unit tests for the RF modeling core module.
Author:      Pessel Arnaud
Date:        2026-05-02
License:     MIT
"""

import matplotlib
import matplotlib.pyplot as plt
import pytest
from numpy import abs, conj

from rf_chain_modeling.rf_utils.rf_modeling import Signals, calculate_spectral_rms_dbm

# Use a non-interactive backend so tests do not block by opening windows
matplotlib.use("Agg")


def test_signal_initialization(empty_signal: Signals) -> None:
    """Verify that the signal is properly initialized with correct dimensions."""
    assert empty_signal.fmax      == 20e9
    assert empty_signal.bandwidth == empty_signal.sampling_rate / 2

    # The base level initiated at -1000 dBm should give an RMS close to -1000 dBm
    assert empty_signal.rms_dbm() == pytest.approx(-1000.0, abs=1.0)


def test_add_tone_rms_power(signal_with_tone: Signals) -> None:
    """Verify that adding a -20 dBm tone results in an overall RMS power of -20 dBm."""
    # Tolerance of 0.1 dBm
    assert signal_with_tone.rms_dbm() == pytest.approx(-20.0, abs=0.1)


def test_thermal_noise_addition(empty_signal: Signals) -> None:
    """Verify that adding thermal noise properly modifies the signal power."""
    power_before = empty_signal.rms_dbm()
    empty_signal.add_thermal_noise()
    power_after = empty_signal.rms_dbm()

    # The signal must be significantly noisier than the -1000 dBm base
    assert power_after > power_before
    assert power_after > -150.0  # Typical thermal noise value depending on bandwidth


def test_plotting_functions_execution(signal_with_tone: Signals) -> None:
    """Verify that visualization functions execute without errors.

    We do not verify the image itself, just ensure that matplotlib code does not crash.
    """
    try:
        signal_with_tone.plot_temporal(tmax=10e-9)
        signal_with_tone.plot_spectrum()
    except Exception as plotting_error:
        pytest.fail(f"Plotting functions raised an exception: {plotting_error}")
    finally:
        # Close the figure to free up memory
        plt.close("all")

def test_complex_multi_tone_signal() -> None:
    """Verify the generation of a complex signal with multiple tones and noise.

    This test replicates the original main() demonstration scenario and mathematically
    validates the RMS power and the spectral peaks at specific frequencies.
    """
    from numpy import pi

    from rf_chain_modeling.rf_utils.rf_modeling import Signals, thermal_noise_power_dbm

    # 1. Setup the exact same signal from the original main()
    signal = Signals(fmax=20e9, bin_width=100e6)
    signal.add_noise(thermal_noise_power_dbm(signal.temp_kelvin, signal.bandwidth))

    # Add tones
    tones = [(17e9, -10, -pi / 3), (11e9, -5, pi / 4), (3e9, 0, 0)]
    for fr, pow, ph in tones:
        signal.add_tone(fr, pow, ph)

    # 2. Verify overall RMS power
    # The dominant power is from the 0 dBm tone. The -50 and -55 dBm tones 
    # contribute a negligible amount of power in linear scale.
    assert signal.rms_dbm() == pytest.approx(1.51, abs=0.1)

    # 3. Verify the FFT spectrum peaks
    for fr, pow, ph in tones:
        idx_neg   = signal.get_arg_freq(-fr)
        idx_pos   = signal.get_arg_freq( fr)
        power_rms = calculate_spectral_rms_dbm(signal.spectrums[:, (idx_neg, idx_pos)].mean(axis=0), signal.imped_ohms)

        # Check if the combined power matches the generated tones
        # (Allowing a 1.0 dBm tolerance due to spectral leakage / windowing effects)
        assert power_rms == pytest.approx(pow, abs=0.1),f"Failed for frequency: {fr} Hz. Obtained power_rms: {power_rms} dBm, expected: {pow} dBm"