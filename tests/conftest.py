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

from rf_chain_modeling.rf_utils.rf_modeling import Signals


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
def noise_signal() -> Signals:
    """Return a Signals object containing pure thermal noise.

    Returns:
        A Signals instance with thermal noise at 290K.
    """
    sig = Signals(fmax=20e9, bin_width=500e6, n_windows=4)
    sig.add_thermal_noise(temp_kelvin=290.0)
    return sig
