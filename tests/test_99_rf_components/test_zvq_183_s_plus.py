#!/usr/bin/env python
# coding: utf-8

"""
Project: rf_chain_modeling
Module: tests.rf_components.test_zvq_183_s_plus
Description: Unit tests for the ZVQ-183-S+ component model.
Author: Pessel Arnaud
Date: 2026-05-05
Version: 0.1.2.dev1
License: MIT
"""

import numpy as np

from rf_chain_modeling.rf_components.zvq_183_s_plus import cpnt, csv_data
from rf_chain_modeling.rf_utils.rf_modeling import RF_Modelised_Component


def test_csv_data_loaded():
    """Verify that the TSV data file is successfully loaded and parsed."""
    assert csv_data is not None

    # On doit utiliser dtype.names pour les numpy structured arrays
    assert "freq" in csv_data.data.dtype.names
    assert "gain" in csv_data.data.dtype.names
    assert len(csv_data.data["freq"]) > 0


def test_component_instantiation():
    """Verify that the component is correctly instantiated with matching array lengths."""
    assert isinstance(cpnt, RF_Modelised_Component)

    freqs_len = len(cpnt.freqs)

    assert len(cpnt.gains_db)   == freqs_len
    assert len(cpnt.phases_rad) == freqs_len
    assert len(cpnt.nfs_db)     == freqs_len
    assert len(cpnt.op1ds_dbm)  == freqs_len


def test_assess_gain():
    """Test the gain assessment method over a small frequency range."""
    fmin = 10e6
    fmax = 20e6
    fstp = 2e6

    freqs, gains, phases, noise_figures = cpnt.assess_gain(
        fmin = fmin, 
        fmax = fmax, 
        fstp = fstp
    )

    assert len(freqs) > 0
    assert len(freqs) == len(gains)
    assert len(freqs) == len(phases)
    assert len(freqs) == len(noise_figures)
    assert np.all(freqs >= fmin)
    assert np.all(freqs <= fmax)


def test_assess_ipx():
    """Test the intercept point assessment method."""
    fmin = 10e6
    fmax = 20e6
    fstp = 2e6

    results = cpnt.assess_ipx(
        fmin = fmin, 
        fmax = fmax, 
        fstp = fstp
    )

    assert results is not None
    assert results.shape[0] > 0
    assert results.shape[1] == 6
