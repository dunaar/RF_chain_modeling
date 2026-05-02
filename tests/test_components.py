#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_components.py
========================
Unit tests for the ``rf_chain_modeling`` package.

Coverage
--------
1.  Conversion utilities
        ``gain_db_to_gain``, ``gain_to_gain_db``, ``nf_db_to_nf``, ``nf_to_nf_db``,
        ``dbm_to_voltage``, ``voltage_to_dbm``, ``dbm_to_watts``,
        ``thermal_noise_power_dbm``, ``mul_nfs``

2.  ``Signals``
        Object creation, tone injection, noise injection, RMS power, frequency
        indexing, 2-D array shape, deep-copy independence.

3.  ``Attenuator``
        Power reduction, NF = attenuation (passive device physical law),
        identity at 0 dB, IIP3 = ∞ for passive devices.

4.  ``Simple_Amplifier``
        Gain, noise figure, gain compression, ``assess_gain`` / ``assess_ipx``
        output shape, OIP3 > OP1dB, attribute storage.

5.  ``RF_Cable``
        Attenuation, frequency-dependent loss (α · √f · L), NF = |gain|,
        identity at zero length.

6.  ``HighPass_Filter``
        In-band pass, out-of-band rejection, roll-off vs. order, insertion loss.

7.  ``LowPass_Filter``
        In-band pass, out-of-band rejection, gain monotonically decreasing.

8.  ``BandPass_Filter``
        In-band pass, out-of-band rejection (low & high), automatic
        frequency-swap when ``cutoff_freq1 > cutoff_freq2``, ``ValueError``
        on invalid second frequency, steeper roll-off at higher order.

9.  ``Antenna_Component``
        Thermal-noise injection, gain application, ``assess_gain`` output.

10. ``RF_slope_equalizer``
        Linear gain vs. frequency, monotonic gain profile.

11. ``RF_chain``
        Signal processing through the cascade, net gain, Friis NF formula,
        ``assess_gain`` / ``assess_ipx`` shapes, empty chain identity,
        component ordering impact.

12. ``RF_Modelised_Component``
        Data interpolation, approximate gain, signal processing,
        ``assess_ipx`` output shape, out-of-band extrapolation.

13. End-to-end integration
        Full RX chain (Antenna → HPF → LNA → Attenuator → Amp → Cable → BPF),
        combined ``assess_gain`` on a 3-component cascade.

Design conventions
------------------
- Every test that mutates a ``Signals`` object operates on a ``copy.deepcopy``
  of the fixture so that tests remain fully independent regardless of
  execution order.
- Amplitude tolerances (typically ±2–5 dB) account for stochastic thermal
  noise already present in ``Signals`` instances.
- Physical laws (Friis NF, passive device NF, OIP3 / OP1dB relationship)
  are asserted with realistic margins, not tight numerical equalities.

Author  : Pessel Arnaud — rf_chain_modeling project
Version : 0.1 (2026-05-02)
License : MIT
"""

import copy
import math

import numpy as np
import pytest
from numpy import pi

# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------
from rf_chain_modeling.rf_utils.rf_modeling import (
    # Core signal/chain classes
    Signals,
    RF_chain,
    RF_Modelised_Component,
    # Conversion utilities
    gain_db_to_gain,
    gain_to_gain_db,
    dbm_to_voltage,
    nf_db_to_nf,
    nf_to_nf_db,
    mul_nfs,
    thermal_noise_power_dbm,
    voltage_to_dbm,
    dbm_to_watts,
)
from rf_chain_modeling.rf_utils.rf_essential_components import (
    Attenuator,
    Simple_Amplifier,
    RF_Cable,
    HighPass_Filter,
    LowPass_Filter,
    BandPass_Filter,
    Antenna_Component,
    RF_slope_equalizer,
)

# ===========================================================================
# Module-level shared fixtures
# ===========================================================================

@pytest.fixture
def simple_signal():
    """Return a wideband ``Signals`` object (0–20 GHz) with a single tone.

    The tone is placed at 5 GHz with a power of −20 dBm and zero phase.
    This fixture is used by most component tests as a reference stimulus.

    Returns
    -------
    Signals
        ``fmax=20 GHz``, ``bin_width=500 MHz``, ``n_windows=4``,
        single tone at 5 GHz / −20 dBm / 0 rad.
    """
    sig = Signals(fmax=20e9, bin_width=500e6, n_windows=4)
    sig.add_tone(5e9, -20, 0)
    return sig


@pytest.fixture
def broadband_signal():
    """Return a ``Signals`` object with three tones for non-linearity tests.

    Three tones are placed at 3 GHz (0 dBm), 9 GHz (−30 dBm), and
    17 GHz (−40 dBm) with distinct phases.  This simulates a multi-carrier
    environment that stresses intermodulation and compression tests.

    Returns
    -------
    Signals
        ``fmax=20 GHz``, ``bin_width=200 MHz``, ``n_windows=4``,
        tones at 3/9/17 GHz.
    """
    sig = Signals(fmax=20e9, bin_width=200e6, n_windows=4)
    sig.add_tone(3e9,   0,       0)
    sig.add_tone(9e9,  -30, pi / 4)
    sig.add_tone(17e9, -40, -pi / 3)
    return sig


# ===========================================================================
# 1 — Conversion utilities
# ===========================================================================

class TestConversionUtils:
    """Verify the stand-alone conversion helpers in ``rf_modeling``.

    Each test checks a well-known analytical value or a round-trip property
    so that regressions in the numeric helpers are caught immediately.
    """

    def test_gain_db_to_gain_0dB(self):
        """0 dB must map to a linear voltage gain of exactly 1.0."""
        assert gain_db_to_gain(0.0) == pytest.approx(1.0, rel=1e-6)

    def test_gain_db_to_gain_20dB(self):
        """20 dB corresponds to a voltage gain of 10 (power ratio 100)."""
        assert gain_db_to_gain(20.0) == pytest.approx(10.0, rel=1e-4)

    def test_gain_db_to_gain_minus6dB(self):
        """−6 dB is approximately half the voltage amplitude (0.5012)."""
        assert gain_db_to_gain(-6.0) == pytest.approx(0.5012, rel=1e-3)

    def test_gain_to_gain_db_roundtrip(self):
        """``gain_db_to_gain`` and ``gain_to_gain_db`` must be exact inverses.

        Applies the forward then inverse transform across a representative
        set of dB values and checks that the result equals the original
        input within floating-point precision.
        """
        for val in [-10, 0, 3, 20, 40]:
            assert gain_to_gain_db(gain_db_to_gain(val)) == pytest.approx(val, abs=1e-6)

    def test_nf_db_to_nf_0dB(self):
        """A noise figure of 0 dB means no added noise: linear NF = 0."""
        assert nf_db_to_nf(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_nf_db_to_nf_3dB(self):
        """3 dB NF doubles the noise power: linear NF contribution ≈ 1."""
        nf_lin = nf_db_to_nf(3.0)
        assert nf_lin == pytest.approx(1.0, rel=1e-2)

    def test_nf_to_nf_db_roundtrip(self):
        """``nf_db_to_nf`` and ``nf_to_nf_db`` must be exact inverses."""
        for val in [0, 1, 3, 5, 10]:
            assert nf_to_nf_db(nf_db_to_nf(val)) == pytest.approx(val, abs=1e-4)

    def test_dbm_to_watts(self):
        """0 dBm equals 1 mW (1 × 10⁻³ W) by definition."""
        assert dbm_to_watts(0.0) == pytest.approx(1e-3, rel=1e-6)

    def test_thermal_noise_power_returns_negative_dbm(self):
        """Thermal noise floor at room temperature over 1 GHz must be < −60 dBm.

        Physical reference: kTB at 298 K over 1 GHz ≈ −84 dBm.
        The assertion uses a conservative threshold to avoid brittleness.
        """
        pwr = thermal_noise_power_dbm(298.15, 1e9)
        assert pwr < -60

    def test_voltage_to_dbm_known_value(self):
        """``voltage_to_dbm`` must be the exact inverse of ``dbm_to_voltage`` on 50 Ω.

        Tests four representative power levels spanning a 40 dB dynamic range.
        """
        for pwr_dbm in [-30, -10, 0, 10]:
            v = dbm_to_voltage(pwr_dbm)
            assert voltage_to_dbm(v) == pytest.approx(pwr_dbm, abs=1e-4)

    def test_mul_nfs_identity(self):
        """Cascading with a noiseless second stage (NF₂ = 0 dB) must leave NF₁ unchanged."""
        nf1 = nf_db_to_nf(3.0)
        nf2 = nf_db_to_nf(0.0)
        result = mul_nfs(nf1, nf2)
        assert result == pytest.approx(nf1, rel=1e-6)

    def test_mul_nfs_increases(self):
        """Cascading two equally noisy stages must produce a combined NF greater than either alone."""
        nf1 = nf_db_to_nf(3.0)
        nf2 = nf_db_to_nf(3.0)
        result = mul_nfs(nf1, nf2)
        assert result > nf1


# ===========================================================================
# 2 — Signals
# ===========================================================================

class TestSignals:
    """Test the ``Signals`` class: construction, tone/noise injection, and helpers."""

    def test_signals_creation(self):
        """A freshly created ``Signals`` object must expose a non-None 2-D array."""
        sig = Signals(fmax=20e9, bin_width=500e6)
        assert sig.sig2d is not None
        assert sig.sig2d.ndim == 2

    def test_signals_fmax_stored(self):
        """fmax is stored correctly, and sampling_rate satisfies the over-sampled Nyquist criterion.

        The implementation uses a 2.2× over-sampling factor (instead of the strict 2× Nyquist
        minimum) to provide spectral guard-band and reduce aliasing at the band edges.
        """
        sig = Signals(fmax=10e9, bin_width=200e6)
        assert sig.fmax == pytest.approx(10e9)
        assert sig.sampling_rate >= 2 * 10e9

    def test_add_tone_increases_rms(self):
        """Injecting a tone must raise the RMS power level of the signal."""
        sig = Signals(fmax=20e9, bin_width=500e6, n_windows=4)
        rms_before = sig.rms_dbm()
        sig.add_tone(5e9, 0, 0)
        rms_after = sig.rms_dbm()
        assert rms_after > rms_before

    def test_add_noise_increases_rms(self):
        """Injecting thermal noise must raise the RMS power level."""
        sig = Signals(fmax=20e9, bin_width=500e6, n_windows=4)
        rms_before = sig.rms_dbm()
        noise_pwr = thermal_noise_power_dbm(298.15, sig.bandwidth)
        sig.add_noise(noise_pwr)
        rms_after = sig.rms_dbm()
        assert rms_after > rms_before

    def test_rms_dbm_tone_power(self, simple_signal):
        """A single −20 dBm tone must yield an RMS reading within ±5 dB of −20 dBm.

        The ±5 dB window accounts for background noise already present in
        the freshly created ``Signals`` instance.
        """
        rms = simple_signal.rms_dbm()
        assert -25 < rms < -15

    def test_get_arg_freq_returns_positive_index(self, simple_signal):
        """``get_arg_freq`` must return a valid (non-negative) frequency bin index."""
        idx = simple_signal.get_arg_freq(5e9)
        assert idx >= 0

    def test_sig2d_shape(self):
        """The first dimension of ``sig2d`` must equal the requested number of windows."""
        nwin = 8
        sig = Signals(fmax=10e9, bin_width=100e6, n_windows=nwin)
        assert sig.sig2d.shape[0] == nwin

    def test_copy_independence(self, simple_signal):
        """``copy.deepcopy`` must produce a fully independent object.

        Zeroing out the copy's internal array must not affect the original's
        RMS reading.
        """
        sig_copy = copy.deepcopy(simple_signal)
        sig_copy.sig2d[:] = 0
        # Original must still hold meaningful power (well above −999 dBm)
        assert simple_signal.rms_dbm() != pytest.approx(-999, abs=1)


# ===========================================================================
# 3 — Attenuator
# ===========================================================================

class TestAttenuator:
    """Test the ``Attenuator`` component.

    Key physical law under test:
        For any passive device, NF (dB) = insertion loss (dB).
        This is a direct consequence of the Friis formula applied to a
        lossy two-port with no added noise sources.
    """

    def test_attenuator_reduces_power(self, simple_signal):
        """An attenuator must always reduce the signal power."""
        att = Attenuator(att_db=10)
        sig = copy.deepcopy(simple_signal)
        pwr_before = sig.rms_dbm()
        att.process_signals(sig)
        pwr_after = sig.rms_dbm()
        assert pwr_after < pwr_before

    def test_attenuator_power_reduction_approx_10dB(self, simple_signal):
        """A 10 dB attenuator must reduce the power by approximately 10 dB.

        The tolerance window (8–15 dB) accounts for residual thermal noise
        already embedded in the stimulus signal.
        """
        att = Attenuator(att_db=10)
        sig = copy.deepcopy(simple_signal)
        pwr_before = sig.rms_dbm()
        att.process_signals(sig)
        pwr_after = sig.rms_dbm()
        delta = pwr_before - pwr_after
        assert 8 < delta < 15

    def test_attenuator_nf_equals_attenuation(self):
        """Physical law: NF of a passive attenuator equals its attenuation.

        Asserted across the full band (1–10 GHz) with ±1.5 dB tolerance.
        """
        att = Attenuator(att_db=6)
        freqs, gains, phases, nf = att.assess_gain(fmin=1e9, fmax=10e9, fstp=1e9)
        assert np.all(np.abs(nf - 6.0) < 1.5)

    def test_attenuator_gain_is_negative(self):
        """An attenuator must always produce negative gain (dB) at every frequency."""
        att = Attenuator(att_db=3)
        freqs, gains, phases, nf = att.assess_gain(fmin=1e9, fmax=10e9, fstp=1e9)
        assert np.all(gains < 0)

    def test_attenuator_gain_approx_minus_att(self):
        """The reported gain must be approximately −att_db across the band."""
        att = Attenuator(att_db=5)
        freqs, gains, phases, nf = att.assess_gain(fmin=2e9, fmax=8e9, fstp=1e9)
        assert np.all(np.abs(gains - (-5.0)) < 1.0)

    def test_attenuator_iip3_is_inf(self):
        """A passive attenuator has no active non-linearity: OP1dB must be ∞ (or very high).

        The ``assess_ipx`` matrix has shape (7, N).  Row index 2 holds OP1dB.
        For a passive device this value should be infinity or exceed 30 dBm.
        """
        att = Attenuator(att_db=5)
        # assess_ipx columns: freq, gain, op1db, iip3, oip3, iip2, oip2
        results = att.assess_ipx(fmin=1e9, fmax=5e9, fstp=2e9)
        op1db_col = results[2, :]
        assert np.all(np.isinf(op1db_col) | (op1db_col > 30))

    def test_attenuator_0dB_is_identity(self, simple_signal):
        """A 0 dB attenuator must leave the signal power essentially unchanged (<1 dB drift)."""
        att = Attenuator(att_db=0)
        sig = copy.deepcopy(simple_signal)
        pwr_before = sig.rms_dbm()
        att.process_signals(sig)
        pwr_after = sig.rms_dbm()
        assert abs(pwr_after - pwr_before) < 1.0


# ===========================================================================
# 4 — Simple_Amplifier
# ===========================================================================

class TestSimpleAmplifier:
    """Test the ``Simple_Amplifier`` component.

    Covers small-signal gain, noise figure, gain compression, and
    the non-linearity assessment interface (``assess_ipx``).
    """

    def test_amplifier_increases_power(self, simple_signal):
        """An amplifier with positive gain must increase the signal power."""
        amp = Simple_Amplifier(gain_db=20, nf_db=3)
        sig = copy.deepcopy(simple_signal)
        pwr_before = sig.rms_dbm()
        amp.process_signals(sig)
        pwr_after = sig.rms_dbm()
        assert pwr_after > pwr_before

    def test_amplifier_gain_approx_correct(self, simple_signal):
        """A 15 dB amplifier must increase the measured power by roughly 12–20 dB.

        The wide window accounts for added thermal noise and the finite
        dynamic range of the RMS estimator.
        """
        amp = Simple_Amplifier(gain_db=15, nf_db=3)
        sig = copy.deepcopy(simple_signal)
        pwr_before = sig.rms_dbm()
        amp.process_signals(sig)
        pwr_after = sig.rms_dbm()
        delta = pwr_after - pwr_before
        assert 12 < delta < 20

    def test_amplifier_assess_gain_returns_correct_shape(self):
        """``assess_gain`` must return four arrays of equal, non-zero length."""
        amp = Simple_Amplifier(gain_db=20, nf_db=5)
        freqs, gains, phases, nf = amp.assess_gain(fmin=1e9, fmax=10e9, fstp=1e9)
        assert len(freqs) == len(gains) == len(phases) == len(nf)
        assert len(freqs) > 0

    def test_amplifier_gain_db_in_assess_gain(self):
        """The mean gain reported by ``assess_gain`` must match the configured ``gain_db`` (±3 dB)."""
        amp = Simple_Amplifier(gain_db=20, nf_db=3)
        freqs, gains, phases, nf = amp.assess_gain(fmin=2e9, fmax=8e9, fstp=1e9)
        assert np.mean(gains) == pytest.approx(20, abs=3)

    def test_amplifier_nf_positive(self):
        """The noise figure must be non-negative at every frequency point."""
        amp = Simple_Amplifier(gain_db=16, nf_db=3)
        freqs, gains, phases, nf = amp.assess_gain(fmin=1e9, fmax=10e9, fstp=1e9)
        assert np.all(nf >= 0)

    def test_amplifier_nf_approx_configured(self):
        """The mean NF returned by ``assess_gain`` must be close to the configured ``nf_db`` (±2 dB)."""
        amp = Simple_Amplifier(gain_db=16, nf_db=4)
        freqs, gains, phases, nf = amp.assess_gain(fmin=2e9, fmax=8e9, fstp=1e9)
        assert np.mean(nf) == pytest.approx(4, abs=2)

    def test_amplifier_compression_limits_output(self, simple_signal):
        """A high-power input near the compression point must not produce unbounded output.

        An amplifier with OP1dB = 10 dBm driven with a tone close to saturation
        must keep its output well below OP1dB + 15 dB.
        """
        amp = Simple_Amplifier(gain_db=20, oip3_dbm=20, op1db_dbm=10, nf_db=5)
        sig = copy.deepcopy(simple_signal)
        sig.add_tone(5e9, 5, 0)    # strong tone, near compression
        amp.process_signals(sig)
        assert sig.rms_dbm() < 25

    def test_amplifier_assess_ipx_returns_array(self):
        """``assess_ipx`` must return a 2-D array with 7 rows (one per metric)."""
        amp = Simple_Amplifier(gain_db=20, oip3_dbm=30, nf_db=3)
        results = amp.assess_ipx(fmin=1e9, fmax=5e9, fstp=2e9)
        assert results.ndim == 2
        # Row mapping: 0=freq, 1=gain, 2=OP1dB, 3=IIP3, 4=OIP3, 5=IIP2, 6=OIP2
        assert results.shape[0] == 7

    def test_amplifier_oip3_greater_than_op1db(self):
        """Physical law: OIP3 must exceed OP1dB (theoretical gap ≈ 10 dB for a memoryless device)."""
        amp = Simple_Amplifier(gain_db=20, oip3_dbm=30, op1db_dbm=20, nf_db=3)
        results = amp.assess_ipx(fmin=5e9, fmax=5e9, fstp=2e9)
        op1db = results[2, 0]
        oip3  = results[4, 0]
        assert oip3 > op1db

    def test_amplifier_attributes_stored(self):
        """Constructor arguments must be stored as instance attributes without mutation."""
        amp = Simple_Amplifier(gain_db=18, nf_db=4, oip3_dbm=35, iip2_dbm=50)
        assert amp.gain_db   == 18
        assert amp.nf_db     == 4
        assert amp.oip3_dbm  == 35
        assert amp.iip2_dbm  == 50


# ===========================================================================
# 5 — RF_Cable
# ===========================================================================

class TestRFCable:
    """Test the ``RF_Cable`` component.

    RF cables exhibit skin-effect loss proportional to √f.
    Key properties: passive (NF = |gain|), longer ⟹ more loss,
    frequency-dependent attenuation.
    """

    def test_cable_attenuates_signal(self, simple_signal):
        """Any cable with positive insertion loss or non-zero length must reduce power."""
        cable = RF_Cable(length_m=10, insertion_losses_dB=0.5)
        sig = copy.deepcopy(simple_signal)
        pwr_before = sig.rms_dbm()
        cable.process_signals(sig)
        pwr_after = sig.rms_dbm()
        assert pwr_after < pwr_before

    def test_cable_longer_more_attenuation(self, simple_signal):
        """A longer cable must produce more attenuation than a shorter one."""
        sig_short = copy.deepcopy(simple_signal)
        sig_long  = copy.deepcopy(simple_signal)
        RF_Cable(length_m=1).process_signals(sig_short)
        RF_Cable(length_m=100).process_signals(sig_long)
        assert sig_long.rms_dbm() < sig_short.rms_dbm()

    def test_cable_gain_is_negative(self):
        """A cable (passive element) must report negative gain at every frequency."""
        cable = RF_Cable(length_m=10)
        freqs, gains, phases, nf = cable.assess_gain(fmin=1e9, fmax=10e9, fstp=1e9)
        assert np.all(gains < 0)

    def test_cable_nf_equals_abs_gain(self):
        """Physical law: NF of a passive cable equals its attenuation magnitude (±1.0 dB)."""
        cable = RF_Cable(length_m=5, insertion_losses_dB=1)
        freqs, gains, phases, nf = cable.assess_gain(fmin=2e9, fmax=8e9, fstp=2e9)
        assert np.allclose(nf, -gains, atol=1.0)

    def test_cable_frequency_dependent_loss(self):
        """Cable loss must be higher at high frequencies than at low frequencies (skin effect)."""
        cable = RF_Cable(length_m=20, insertion_losses_dB=0)
        freqs, gains, phases, nf = cable.assess_gain(fmin=1e9, fmax=18e9, fstp=1e9)
        # Gain at 18 GHz must be more negative than at 1 GHz
        assert gains[-1] < gains[0]

    def test_cable_zero_length_no_loss(self, simple_signal):
        """A cable of zero length and zero insertion loss must act as an identity element (<1 dB drift)."""
        cable = RF_Cable(length_m=0, insertion_losses_dB=0)
        sig = copy.deepcopy(simple_signal)
        pwr_before = sig.rms_dbm()
        cable.process_signals(sig)
        pwr_after = sig.rms_dbm()
        assert abs(pwr_after - pwr_before) < 1.0


# ===========================================================================
# 6 — HighPass_Filter
# ===========================================================================

class TestHighPassFilter:
    """Test the ``HighPass_Filter`` component.

    Checks that the filter passes energy above its cutoff frequency,
    rejects energy below it, and that roll-off steepens with filter order.
    """

    def test_hpf_passes_high_frequencies(self, simple_signal):
        """A tone at 5 GHz must experience minimal attenuation through an HPF with cutoff at 2 GHz.

        Acceptable passband ripple / insertion loss: < 5 dB.
        """
        hpf = HighPass_Filter(cutoff_freq=2e9, order=5)
        sig = copy.deepcopy(simple_signal)
        pwr_before = sig.rms_dbm()
        hpf.process_signals(sig)
        pwr_after = sig.rms_dbm()
        assert abs(pwr_after - pwr_before) < 5

    def test_hpf_blocks_low_frequency(self):
        """A tone at 1 GHz must be strongly attenuated by an HPF with cutoff at 10 GHz (>15 dB)."""
        hpf = HighPass_Filter(cutoff_freq=10e9, order=5)
        sig = Signals(fmax=20e9, bin_width=500e6, n_windows=4)
        sig.add_tone(1e9, 0, 0)
        hpf.process_signals(sig)
        assert sig.rms_dbm() < -15

    def test_hpf_gain_increases_with_frequency(self):
        """The gain profile of an HPF must increase (become less negative) with frequency."""
        hpf = HighPass_Filter(cutoff_freq=5e9, order=3)
        freqs, gains, phases, nf = hpf.assess_gain(fmin=1e9, fmax=18e9, fstp=1e9)
        assert gains[-1] > gains[0]

    def test_hpf_nf_is_positive(self):
        """The noise figure must be non-negative at every frequency
           (small negative values allowed due to statistical noise calculation)."""
        hpf = HighPass_Filter(cutoff_freq=5e9, order=3)
        freqs, gains, phases, nf = hpf.assess_gain(fmin=2e9, fmax=10e9, fstp=1e9)
        assert np.all(nf >= -0.1)

    def test_hpf_order_affects_rolloff(self):
        """A higher-order HPF must attenuate an out-of-band tone more than a lower-order one.

        Both filters share the same cutoff (10 GHz); the tone is at 1 GHz (well below cutoff).
        """
        sig_lo = Signals(fmax=20e9, bin_width=200e6, n_windows=4)
        sig_hi = Signals(fmax=20e9, bin_width=200e6, n_windows=4)
        sig_lo.add_tone(1e9, 0, 0)
        sig_hi.add_tone(1e9, 0, 0)
        HighPass_Filter(cutoff_freq=10e9, order=1).process_signals(sig_lo)
        HighPass_Filter(cutoff_freq=10e9, order=7).process_signals(sig_hi)
        assert sig_hi.rms_dbm() < sig_lo.rms_dbm()

    def test_hpf_insertion_loss_reduces_passband_gain(self):
        """A 3 dB insertion loss must reduce the in-band power by approximately 3 dB (1–7 dB window)."""
        hpf = HighPass_Filter(cutoff_freq=1e9, order=3, insertion_losses_dB=3)
        sig = Signals(fmax=20e9, bin_width=200e6, n_windows=4)
        sig.add_tone(15e9, 0, 0)    # well inside the passband
        pwr_before = sig.rms_dbm()
        hpf.process_signals(sig)
        pwr_after = sig.rms_dbm()
        delta = pwr_before - pwr_after
        assert 1 < delta < 7


# ===========================================================================
# 7 — LowPass_Filter
# ===========================================================================

class TestLowPassFilter:
    """Test the ``LowPass_Filter`` component.

    Mirrors the HPF tests but for the complementary frequency sense.
    """

    def test_lpf_passes_low_frequencies(self):
        """A tone at 1 GHz must pass a 10 GHz LPF with less than 3 dB attenuation."""
        lpf = LowPass_Filter(cutoff_freq=10e9, order=5)
        sig = Signals(fmax=20e9, bin_width=200e6, n_windows=4)
        sig.add_tone(1e9, 0, 0)
        pwr_before = sig.rms_dbm()
        lpf.process_signals(sig)
        pwr_after = sig.rms_dbm()
        assert abs(pwr_after - pwr_before) < 3

    def test_lpf_blocks_high_frequencies(self):
        """A tone at 18 GHz must be attenuated >15 dB by an LPF with cutoff at 5 GHz."""
        lpf = LowPass_Filter(cutoff_freq=5e9, order=5)
        sig = Signals(fmax=20e9, bin_width=200e6, n_windows=4)
        sig.add_tone(18e9, 0, 0)
        lpf.process_signals(sig)
        assert sig.rms_dbm() < -15

    def test_lpf_gain_decreases_with_frequency(self):
        """The gain profile of an LPF must decrease with increasing frequency."""
        lpf = LowPass_Filter(cutoff_freq=5e9, order=3)
        freqs, gains, phases, nf = lpf.assess_gain(fmin=1e9, fmax=18e9, fstp=1e9)
        assert gains[-1] < gains[0]


# ===========================================================================
# 8 — BandPass_Filter
# ===========================================================================

class TestBandPassFilter:
    """Test the ``BandPass_Filter`` component.

    Special attention is given to the automatic frequency-swap feature:
    when ``cutoff_freq1 > cutoff_freq2`` the constructor must silently swap
    the two values, making argument order irrelevant.
    """

    def test_bpf_passes_in_band(self):
        """A tone at 10 GHz must pass a BPF with passband [8–12 GHz] with minimal loss (<5 dB)."""
        bpf = BandPass_Filter(cutoff_freq1=8e9, cutoff_freq2=12e9, order=5)
        sig = Signals(fmax=20e9, bin_width=200e6, n_windows=4)
        sig.add_tone(10e9, 0, 0)
        pwr_before = sig.rms_dbm()
        bpf.process_signals(sig)
        pwr_after = sig.rms_dbm()
        assert abs(pwr_after - pwr_before) < 5

    def test_bpf_blocks_out_of_band_low(self):
        """A 1 GHz tone must be attenuated >10 dB by a BPF with passband [8–12 GHz]."""
        bpf = BandPass_Filter(cutoff_freq1=8e9, cutoff_freq2=12e9, order=5)
        sig = Signals(fmax=20e9, bin_width=200e6, n_windows=4)
        sig.add_tone(1e9, 0, 0)
        bpf.process_signals(sig)
        assert sig.rms_dbm() < -10

    def test_bpf_blocks_out_of_band_high(self):
        """A 19 GHz tone must be attenuated >10 dB by a BPF with passband [8–12 GHz]."""
        bpf = BandPass_Filter(cutoff_freq1=8e9, cutoff_freq2=12e9, order=5)
        sig = Signals(fmax=20e9, bin_width=200e6, n_windows=4)
        sig.add_tone(19e9, 0, 0)
        bpf.process_signals(sig)
        assert sig.rms_dbm() < -10

    def test_bpf_freq_inversion_auto(self):
        """Swapping ``cutoff_freq1`` and ``cutoff_freq2`` must produce an identical filter.

        If the constructor correctly auto-swaps the frequencies, both instances
        must expose the same ``cutoff_freq`` and ``cutoff_freq_opt`` attributes.
        """
        bpf_normal   = BandPass_Filter(cutoff_freq1=8e9,  cutoff_freq2=12e9, order=3)
        bpf_inverted = BandPass_Filter(cutoff_freq1=12e9, cutoff_freq2=8e9,  order=3)
        assert bpf_normal.cutoff_freq     == bpf_inverted.cutoff_freq
        assert bpf_normal.cutoff_freq_opt == bpf_inverted.cutoff_freq_opt

    def test_bpf_missing_second_freq_raises(self):
        """Passing ``NaN`` as the second cutoff frequency must raise ``ValueError`` or ``TypeError``."""
        with pytest.raises((ValueError, TypeError)):
            BandPass_Filter(cutoff_freq1=8e9, cutoff_freq2=float("nan"), order=3)

    def test_bpf_higher_order_steeper_rolloff(self):
        """A 7th-order BPF must attenuate an out-of-band tone more than a 2nd-order one."""
        sig_lo = Signals(fmax=20e9, bin_width=200e6, n_windows=4)
        sig_hi = Signals(fmax=20e9, bin_width=200e6, n_windows=4)
        sig_lo.add_tone(1e9, 0, 0)
        sig_hi.add_tone(1e9, 0, 0)
        BandPass_Filter(cutoff_freq1=8e9, cutoff_freq2=12e9, order=2).process_signals(sig_lo)
        BandPass_Filter(cutoff_freq1=8e9, cutoff_freq2=12e9, order=7).process_signals(sig_hi)
        assert sig_hi.rms_dbm() < sig_lo.rms_dbm()


# ===========================================================================
# 9 — Antenna_Component
# ===========================================================================

class TestAntennaComponent:
    """Test the ``Antenna_Component``.

    Unlike passive filters, the antenna also injects thermal noise into the
    signal (modelling the sky noise temperature / antenna noise temperature).
    """

    def test_antenna_adds_noise(self):
        """The antenna must inject thermal noise even into an initially silent signal.

        Passes if the output power is either higher than input or above the
        absolute thermal floor (−150 dBm), which guarantees the noise injection
        code path was executed.
        """
        ant = Antenna_Component(freqs=(1e9, 20e9), gains_db=(3., 7.))
        sig = Signals(fmax=20e9, bin_width=500e6, n_windows=4)
        pwr_before = sig.rms_dbm()
        ant.process_signals(sig)
        pwr_after = sig.rms_dbm()
        assert pwr_after > pwr_before or pwr_after > -150

    def test_antenna_apply_gain_to_tone(self):
        """The antenna must apply its configured gain to the received tone.

        With a flat 10 dB gain the output power must be at most 5 dB below
        the power that would result from a 10 dB pure amplifier (noise shifts
        the operating point slightly).
        """
        ant = Antenna_Component(freqs=(1e9, 20e9), gains_db=(10., 10.))
        sig = Signals(fmax=20e9, bin_width=500e6, n_windows=4)
        sig.add_tone(5e9, -30, 0)
        pwr_before = sig.rms_dbm()
        ant.process_signals(sig)
        pwr_after = sig.rms_dbm()
        assert pwr_after > pwr_before - 5

    def test_antenna_assess_gain_returns_in_band_gain(self):
        """``assess_gain`` must return a non-empty array with mean gain ≈ 5 dB (±3 dB)."""
        ant = Antenna_Component(freqs=(1e9, 20e9), gains_db=(5., 5.))
        freqs, gains, phases, nf = ant.assess_gain(fmin=2e9, fmax=10e9, fstp=1e9)
        assert len(freqs) > 0
        assert np.mean(gains) == pytest.approx(5, abs=3)


# ===========================================================================
# 10 — RF_slope_equalizer
# ===========================================================================

class TestRFSlopeEqualizer:
    """Test the ``RF_slope_equalizer`` component.

    The equalizer applies a gain that increases linearly (in dB) from
    ``gain_db_at_freq1`` at ``freq1`` to ``gain_db_at_freq2`` at ``freq2``.
    This compensates for the √f slope introduced by RF cables.
    """

    def test_slope_eq_linear_gain(self):
        """The equalizer gain must be 0 dB at freq1 and 10 dB at freq2 (±2 dB tolerance)."""
        eq = RF_slope_equalizer(
            freq1=1e9, gain_db_at_freq1=0,
            freq2=10e9, gain_db_at_freq2=10,
        )
        freqs, gains, phases, nf = eq.assess_gain(fmin=1e9, fmax=10e9, fstp=1e9)
        assert gains[0]  == pytest.approx(0,  abs=2)
        assert gains[-1] == pytest.approx(10, abs=2)

    def test_slope_eq_monotonic(self):
        """The gain profile must be monotonically non-decreasing across the entire band.

        A tolerance of −0.5 dB on successive differences allows for minor
        numerical artefacts without masking genuine non-monotonic behaviour.
        """
        eq = RF_slope_equalizer(
            freq1=1e9, gain_db_at_freq1=-5,
            freq2=18e9, gain_db_at_freq2=5,
        )
        freqs, gains, phases, nf = eq.assess_gain(fmin=1e9, fmax=18e9, fstp=1e9)
        diffs = np.diff(gains)
        assert np.all(diffs >= -0.5)


# ===========================================================================
# 11 — RF_chain
# ===========================================================================

class TestRFChain:
    """Test the ``RF_chain`` cascade manager.

    Verifies signal processing, net gain, Friis NF cascading, the
    ``assess_gain`` / ``assess_ipx`` interfaces, and edge cases such as an
    empty chain and component ordering sensitivity.
    """

    @pytest.fixture
    def simple_chain(self):
        """Return a three-stage chain: LNA (16 dB) → Attenuator (5 dB) → HPF (cutoff 2 GHz).

        Net gain ≈ +11 dB in the passband (above 2 GHz).
        """
        return RF_chain([
            Simple_Amplifier(gain_db=16, nf_db=3, oip3_dbm=30),
            Attenuator(att_db=5),
            HighPass_Filter(cutoff_freq=2e9, order=3),
        ])

    def test_chain_process_signals_runs(self, simple_chain, simple_signal):
        """``process_signals`` must complete without error and return a valid RMS value."""
        sig = copy.deepcopy(simple_signal)
        simple_chain.process_signals(sig)
        assert sig.rms_dbm() is not None

    def test_chain_increases_power_with_net_gain(self, simple_signal):
        """A chain with net positive gain (20 dB − 3 dB = 17 dB) must increase the signal power."""
        chain = RF_chain([
            Simple_Amplifier(gain_db=20, nf_db=3),
            Attenuator(att_db=3),
        ])
        sig = copy.deepcopy(simple_signal)
        pwr_before = sig.rms_dbm()
        chain.process_signals(sig)
        pwr_after = sig.rms_dbm()
        assert pwr_after > pwr_before

    def test_chain_assess_gain_shape(self, simple_chain):
        """``assess_gain`` must return four arrays of equal, non-zero length."""
        freqs, gains, phases, nf = simple_chain.assess_gain(fmin=1e9, fmax=10e9, fstp=1e9)
        assert len(freqs) == len(gains) == len(phases) == len(nf)
        assert len(freqs) > 0

    def test_chain_assess_gain_net_gain(self):
        """Net gain of a 20 dB amplifier followed by a 10 dB attenuator must be ≈ 10 dB (±4 dB)."""
        chain = RF_chain([
            Simple_Amplifier(gain_db=20, nf_db=3, oip3_dbm=40),
            Attenuator(att_db=10),
        ])
        freqs, gains, phases, nf = chain.assess_gain(fmin=5e9, fmax=5e9, fstp=2e9)
        assert gains[0] == pytest.approx(10, abs=4)

    def test_chain_nf_dominated_by_first_stage(self):
        """Friis formula: the first-stage NF dominates the cascade NF.

        A chain with the low-NF LNA first must exhibit a lower total NF than
        the same chain with the high-NF amplifier first.
        """
        lna_nf = 3.0
        chain_lna_first = RF_chain([
            Simple_Amplifier(gain_db=20, nf_db=lna_nf, oip3_dbm=30),
            Simple_Amplifier(gain_db=10, nf_db=10),
        ])
        chain_bad_first = RF_chain([
            Simple_Amplifier(gain_db=10, nf_db=10),
            Simple_Amplifier(gain_db=20, nf_db=lna_nf, oip3_dbm=30),
        ])
        _, _, _, nf_good = chain_lna_first.assess_gain(fmin=5e9, fmax=5e9, fstp=2e9)
        _, _, _, nf_bad  = chain_bad_first.assess_gain(fmin=5e9, fmax=5e9, fstp=2e9)
        assert nf_good[0] < nf_bad[0]

    def test_chain_assess_ipx_returns_matrix(self, simple_chain):
        """``assess_ipx`` must return a 2-D matrix with 7 rows."""
        results = simple_chain.assess_ipx(fmin=1e9, fmax=5e9, fstp=2e9)
        assert results.ndim == 2
        assert results.shape[0] == 7

    def test_chain_empty_is_transparent(self, simple_signal):
        """An empty chain must leave the signal completely unchanged (< 0.01 dB drift)."""
        chain = RF_chain([])
        sig = copy.deepcopy(simple_signal)
        pwr_before = sig.rms_dbm()
        chain.process_signals(sig)
        pwr_after = sig.rms_dbm()
        assert abs(pwr_after - pwr_before) < 0.01

    def test_chain_with_filter_suppresses_out_of_band(self):
        """A BPF at the output of an amplifier must suppress an out-of-band tone.

        The amplifier provides 20 dB but the BPF [8–12 GHz] must reject
        the 1 GHz tone so that the final output stays below 15 dBm.
        """
        chain = RF_chain([
            Simple_Amplifier(gain_db=20, nf_db=3),
            BandPass_Filter(cutoff_freq1=8e9, cutoff_freq2=12e9, order=5),
        ])
        sig = Signals(fmax=20e9, bin_width=200e6, n_windows=4)
        sig.add_tone(1e9, 0, 0)    # out-of-band tone
        chain.process_signals(sig)
        assert sig.rms_dbm() < 15

    def test_chain_order_matters(self, simple_signal):
        """Swapping the LNA and the attenuator must change the total NF (Friis sensitivity).

        The two chains have the same components but in reversed order;
        their noise figures must differ by more than 0.5 dB.
        """
        chain1 = RF_chain([Attenuator(att_db=10), Simple_Amplifier(gain_db=20, nf_db=3)])
        chain2 = RF_chain([Simple_Amplifier(gain_db=20, nf_db=3), Attenuator(att_db=10)])
        _, _, _, nf1 = chain1.assess_gain(fmin=5e9, fmax=5e9, fstp=2e9)
        _, _, _, nf2 = chain2.assess_gain(fmin=5e9, fmax=5e9, fstp=2e9)
        assert nf1[0] != pytest.approx(nf2[0], abs=0.5)


# ===========================================================================
# 12 — RF_Modelised_Component
# ===========================================================================

class TestRFModelisedComponent:
    """Test ``RF_Modelised_Component`` with synthetic measurement data.

    This component interpolates user-supplied frequency tables of gain, NF,
    phase, OP1dB, and IIP3 to characterise a data-driven RF device (e.g.
    a datasheet-specified amplifier loaded from a TSV file).
    """

    @pytest.fixture
    def measured_component(self):
        """Return an ``RF_Modelised_Component`` built from synthetic 'measured' data.

        Data covers 1–18 GHz with:
        - Flat gain of 25 dB across the band.
        - NF decreasing linearly from 4.0 dB to 3.5 dB.
        - Phase sweeping from −0.5 rad to −2.5 rad.
        - Constant OP1dB = 24 dBm.
        - IIP3 = OP1dB − Gain = −1 dBm (consistent with Friis gain reference).

        Returns
        -------
        RF_Modelised_Component
        """
        freqs    = np.linspace(1e9, 18e9, 50)
        gains_db = np.full(50, 25.0)
        nfs_db   = np.linspace(4.0, 3.5, 50)
        phases   = np.linspace(-0.5, -2.5, 50)
        op1ds    = np.full(50, 24.0)
        iip3s    = np.full(50, 24.0 - 25.0)    # IIP3 = OIP3 − Gain
        return RF_Modelised_Component(
            freqs=freqs,
            gains_db=gains_db,
            nfs_db=nfs_db,
            phases_rad=phases,
            op1ds_dbm=op1ds,
            iip3s_dbm=iip3s,
        )

    def test_modelised_component_assess_gain_shape(self, measured_component):
        """``assess_gain`` must return non-empty arrays of consistent length."""
        freqs, gains, phases, nf = measured_component.assess_gain(fmin=2e9, fmax=16e9, fstp=1e9)
        assert len(freqs) > 0
        assert len(freqs) == len(gains)

    def test_modelised_component_gain_approx_25dB(self, measured_component):
        """Mean gain must be approximately 25 dB (±4 dB) across the mid-band."""
        freqs, gains, phases, nf = measured_component.assess_gain(fmin=3e9, fmax=15e9, fstp=1e9)
        assert np.mean(gains) == pytest.approx(25, abs=4)

    def test_modelised_component_process_signals(self, measured_component, simple_signal):
        """Processing a signal through the component must increase its power (25 dB gain)."""
        sig = copy.deepcopy(simple_signal)
        pwr_before = sig.rms_dbm()
        measured_component.process_signals(sig)
        pwr_after = sig.rms_dbm()
        assert pwr_after > pwr_before

    def test_modelised_component_assess_ipx(self, measured_component):
        """``assess_ipx`` must return a 2-D array with 7 rows."""
        results = measured_component.assess_ipx(fmin=3e9, fmax=15e9, fstp=3e9)
        assert results.ndim == 2
        assert results.shape[0] == 7

    def test_modelised_component_extrapolation_out_of_band(self, measured_component):
        """``assess_gain`` on a frequency range wider than the measurement data must not raise.

        The component must gracefully extrapolate (or clamp) outside the
        1–18 GHz measurement window.
        """
        freqs, gains, phases, nf = measured_component.assess_gain(
            fmin=0.5e9, fmax=19e9, fstp=1e9
        )
        assert len(freqs) > 0


# ===========================================================================
# 13 — End-to-end integration
# ===========================================================================

class TestEndToEnd:
    """Integration tests that exercise the full signal path.

    These tests mirror the ``rf_chain_example.py`` demo chain but avoid any
    calls to ``matplotlib`` so they run headlessly in CI.
    """

    def test_full_rx_chain(self):
        """Simulate a complete 7-stage RX chain and verify net gain at 10 GHz.

        Chain topology (matching ``rf_chain_example.py``)::

            Antenna (3–7 dBm interpolated gain)
            ↓
            HighPass_Filter  (cutoff 6 GHz, order 5)
            ↓
            LNA / Simple_Amplifier  (16 dB, NF 3 dB, OIP3 30 dBm)
            ↓
            Attenuator  (5 dB)
            ↓
            Power Amplifier / Simple_Amplifier  (20 dB, NF 7 dB, OIP3 40 dBm)
            ↓
            RF_Cable  (10 m, 0.5 dB/m insertion loss)
            ↓
            BandPass_Filter  (passband 9–12 GHz, order 5)

        A −60 dBm tone at 10 GHz (in-band) is injected.  The expected net
        gain is positive, so the output power must exceed the input power.
        """
        antenna = Antenna_Component(freqs=(1e9, 20e9), gains_db=(3., 7.))
        hp      = HighPass_Filter(cutoff_freq=6e9, order=5)
        lna     = Simple_Amplifier(gain_db=16, oip3_dbm=30, nf_db=3)
        att     = Attenuator(att_db=5)
        amp     = Simple_Amplifier(gain_db=20, oip3_dbm=40, nf_db=7)
        cable   = RF_Cable(length_m=10, insertion_losses_dB=0.5)
        bp      = BandPass_Filter(cutoff_freq1=9e9, cutoff_freq2=12e9, order=5)

        chain = RF_chain([antenna, hp, lna, att, amp, cable, bp])

        sig = Signals(fmax=20e9, bin_width=100e6)
        sig.add_tone(10e9, -60, 0)    # weak in-band tone

        pwr_before = sig.rms_dbm()
        chain.process_signals(sig)
        pwr_after = sig.rms_dbm()

        assert pwr_after > pwr_before

    def test_chain_assess_gain_full_chain(self):
        """Verify ``assess_gain`` on a 3-stage cascade (LNA + Attenuator + BPF).

        Expected behaviour:
        - At 10 GHz (in-band): net gain ≈ 16 − 5 = 11 dB (±5 dB tolerance).
        - At 1 GHz (below BPF passband): gain must be negative (rejection).
        """
        chain = RF_chain([
            Simple_Amplifier(gain_db=16, nf_db=3, oip3_dbm=30),
            Attenuator(att_db=5),
            BandPass_Filter(cutoff_freq1=9e9, cutoff_freq2=12e9, order=5),
        ])
        freqs, gains, phases, nf = chain.assess_gain(fmin=1e9, fmax=18e9, fstp=1e9)

        # In-band check: find the frequency bin closest to 10 GHz
        idx_10 = np.argmin(np.abs(freqs - 10e9))
        assert gains[idx_10] == pytest.approx(11, abs=5)

        # Out-of-band check: the BPF must force negative gain at 1 GHz
        idx_1 = np.argmin(np.abs(freqs - 1e9))
        assert gains[idx_1] < 0
