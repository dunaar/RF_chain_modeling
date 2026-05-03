#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project:     rf-chain-modeling
Module:      tests.conftest
Description: Shared pytest fixtures for unit tests.
Author:      Pessel Arnaud
Date:        2026-05-02
License:     MIT
GitHub:      https://github.com/dunaar/RF_chain_modeling
"""

import pytest
from math import pi
from rf_chain_modeling.rf_utils.rf_modeling import Signals


@pytest.fixture
def empty_signal() -> Signals:
    """Return a Signals object without any added tones or external noise.

    Returns:
        A Signals instance initialized for basic empty testing.
    """
    return Signals(fmax=20e9, bin_width=500e6, n_windows=4)


@pytest.fixture
def simple_signal() -> Signals:
    """Return a Signals object with a single 20 dBm tone at 5 GHz.

    Returns:
        A Signals instance initialized for basic component testing.
    """
    sig = Signals(fmax=20e9, bin_width=500e6, n_windows=4)
    sig.add_tone(freq=5e9, power_dbm=-20, phase=0)
    return sig


@pytest.fixture
def signal_with_tone() -> Signals:
    """Alias for simple_signal to accommodate different test namings."""
    sig = Signals(fmax=20e9, bin_width=500e6, n_windows=4)
    sig.add_tone(freq=5e9, power_dbm=-20, phase=0)
    return sig


@pytest.fixture
def noise_signal() -> Signals:
    """Return a Signals object containing pure thermal noise.

    Returns:
        A Signals instance with thermal noise at 290K.
    """
    sig = Signals(fmax=20e9, bin_width=500e6, n_windows=4)
    sig.add_thermal_noise(temp_kelvin=290.0)
    return sig


@pytest.fixture
def broadband_signal() -> Signals:
    """Return a Signals object with three tones for non-linearity tests.

    Three tones are placed at 3 GHz (0 dBm), 9 GHz (-30 dBm), and 17 GHz (-40 dBm) 
    with distinct phases. This simulates a multi-carrier environment.

    Returns:
        Signals (fmax=20 GHz, bin_width=200 MHz, n_windows=4, tones at 3/9/17 GHz).
    """
    sig = Signals(fmax=20e9, bin_width=200e6, n_windows=4)
    sig.add_tone(3e9, 0, 0)
    sig.add_tone(9e9, -30, pi / 4)
    sig.add_tone(17e9, -40, -pi / 3)
    return sig