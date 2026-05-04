#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project:     rf-chain-modeling
Module:      rf_chain_modeling.rf_utils.rf_modeling
Description: RF Signal Simulation and Analysis Framework.
Author:      Pessel Arnaud
Date:        2026-05-02
License:     MIT

This module provides a comprehensive framework for simulating and analyzing RF (Radio Frequency) signals and components.
It includes classes and functions for signal generation, processing, and visualization.

Key Features:
- Signal generation with tones and noise.
- RF component modeling (attenuators, amplifiers, cables, filters, antennas).
- Signal processing and analysis.
- Visualization of temporal and spectral representations.
"""

import copy
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from math import pi

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .util_search_mediane import search_mediane_for_slope

logger = logging.getLogger(__name__)

# ====================================================================================================
# Constants
# ====================================================================================================
# Physical Constants
K_B = 1.38e-23  # Boltzmann constant in Joules per Kelvin

# Default Parameters
DEFAULT_TEMP_KELVIN = 298.15  # Default temperature in Kelvin (298,15K = 25°C)
DEFAULT_IMPED_OHMS  =  50.0   # Default impedance in ohms for RF systems
DEFAULT_N_WINDOWS   =  32     # Default number of signal windows for processing

# ====================================================================================================
# Utility Functions
# ====================================================================================================
def infs_like(arr: np.ndarray) -> np.ndarray:
    """Create an array of the same shape as arr filled with infinity.

    Args:
        arr: Input array to match shape.

    Returns:
        Array filled with infinity.
    """
    return np.full_like(arr, np.inf)

def dbm_to_watts(power_dbm: float | np.ndarray) -> float | np.ndarray:
    """Convert power from dBm to watts.

    Args:
        power_dbm: Power in dBm (decibels relative to 1 milliwatt).

    Returns:
        Power in watts.
    """
    return 10 ** (power_dbm / 10) / 1000  # Convert dBm to milliwatts, then to watts

def watts_to_voltage(power_watts: float | np.ndarray, imped_ohms: float = DEFAULT_IMPED_OHMS) -> float | np.ndarray:
    """Convert power from watts to voltage assuming a given impedance.

    Args:
        power_watts: Power in watts.
        imped_ohms: Impedance in ohms, defaults to 50 ohms.

    Returns:
        Voltage.
    """
    return np.sqrt(power_watts * imped_ohms)  # V = sqrt(P * R) based on Ohm's law

def dbm_to_voltage(power_dbm: float | np.ndarray, imped_ohms: float = DEFAULT_IMPED_OHMS) -> float | np.ndarray:
    """Convert power from dBm to voltage.

    Args:
        power_dbm: Power in dBm.
        imped_ohms: Impedance in ohms, defaults to 50 ohms.

    Returns:
        Voltage.
    """
    return watts_to_voltage(dbm_to_watts(power_dbm), imped_ohms)  # Chain conversion: dBm -> watts -> voltage

def gain_db_to_gain(gain_db: float | np.ndarray) -> float | np.ndarray:
    """Convert gain from dB to linear scale (voltage gain).

    Args:
        gain_db: Gain in decibels.

    Returns:
        Linear voltage gain.
    """
    return 10 ** (gain_db / 20.0)  # Voltage gain: G = 10^(dB/20)

def gain_to_gain_db(gain: float | np.ndarray) -> float | np.ndarray:
    """Convert gain from linear scale to dB.

    Args:
        gain: Linear gain value (scalar or array).

    Returns:
        Gain in dB.
    """
    if isinstance(gain, np.ndarray):
        # In-place modification for arrays (does not create a new copy in memory)
        if np.any(gain == 0):
            gain[gain == 0] = 1e-100
    else:
        # Standard reassignment for scalars (floats are immutable)
        if gain == 0:
            gain = 1e-100

    return 20 * np.log10(np.abs(gain))  # dB = 20 * log10(|G|) for voltage gain

def nf_db_to_nf(nf_db: float | np.ndarray) -> float | np.ndarray:
    """Convert noise figure from dB to linear scale.

    Args:
        nf_db: Noise figure in dB.

    Returns:
        Linear noise figure contribution (standard deviation of noise voltage).
    """
    # Clamp to 0 dB floor: NF < 0 dB is physically impossible for a passive device
    return np.sqrt(np.maximum(10 ** (nf_db / 10) - 1, 0.0))

def nf_to_nf_db(nf: float | np.ndarray) -> float | np.ndarray:
    """Convert noise figure from linear scale to dB.

    Args:
        nf: Linear noise figure (scalar or array).

    Returns:
        Noise figure in dB.
    """
    return 10 * np.log10(nf ** 2 + 1)  # NF_dB = 10 * log10(F), where F = NF_linear^2 + 1

def mul_nfs(nf1: float | np.ndarray, nf2: float | np.ndarray) -> float | np.ndarray:
    """Multiply noise factors to compute combined noise figure using Friis formula.

    Args:
        nf1: Linear noise figure of first component.
        nf2: Linear noise figure of second component.

    Returns:
        Combined linear noise figure.
    """
    return np.sqrt((nf1**2 + 1) * (nf2**2 + 1) - 1)  # Friis formula for noise factor multiplication

def voltage_to_watts(voltage: float | np.ndarray, imped_ohms: float = 50) -> float | np.ndarray:
    """Convert voltage to power in watts.

    Args:
        voltage: Voltage value (scalar or array).
        imped_ohms: Impedance in ohms, defaults to 50 ohms.

    Returns:
        Power in watts.
    """
    return voltage**2 / imped_ohms  # P = V^2 / R

def watts_to_dbm(power_watts: float | np.ndarray) -> float | np.ndarray:
    """Convert power from watts to dBm.

    Args:
        power_watts: Power in watts (scalar or array).

    Returns:
        Power in dBm.
    """
    if isinstance(power_watts, np.ndarray):
        # In-place modification for arrays (does not create a new copy in memory)
        if np.any(power_watts == 0):
            power_watts[power_watts == 0] = 1e-100
    else:
        # Standard reassignment for scalars (floats are immutable)
        if power_watts == 0:
            power_watts = 1e-100

    return 10 * np.log10(power_watts * 1000)  # dBm = 10 * log10(P * 1000) to convert watts to milliwatts

def voltage_to_dbm(voltage: float | np.ndarray, imped_ohms: float = DEFAULT_IMPED_OHMS) -> float | np.ndarray:
    """Convert voltage to power in dBm.

    Args:
        voltage: RMS voltage.
        imped_ohms: Impedance in ohms, defaults to 50 ohms.

    Returns:
        Power in dBm.
    """
    return watts_to_dbm(voltage_to_watts(voltage, imped_ohms))  # Chain conversion: voltage -> watts -> dBm

def calculate_temporal_rms(signal: np.ndarray) -> float:
    """Calculate the Root Mean Square (RMS) value of a temporal signal.

    For temporal signals, the RMS is calculated as the square root of the MEAN of the squared magnitudes of the spectrums.

    Args:
        signal: Input temporal signal as a 1D or 2D array.

    Returns:
        RMS value of the temporal signal.
    """
    return float(np.sqrt(np.mean(np.abs(signal) ** 2)))

def calculate_temporal_rms_dbm(signal: np.ndarray, imped_ohms: float = DEFAULT_IMPED_OHMS) -> float:
    """Calculate the RMS value of a signal in dBm.

    For temporal signals, the RMS is calculated as the square root of the MEAN of the squared magnitudes of the spectrums.

    Args:
        signal: Input signal (1D or 2D array).
        imped_ohms: Impedance in ohms.

    Returns:
        RMS power in dBm.
    """
    return float( voltage_to_dbm(calculate_temporal_rms(signal), imped_ohms) )  # Convert RMS voltage to dBm

def calculate_spectral_rms(spectrums: np.ndarray) -> float:
    """Calculate the Root Mean Square (RMS) value of a spectral signal.

    For spectral signals, the RMS is calculated as the square root of the SUM of the squared magnitudes of the spectrums.

    Args:
        spectrums: Input spectral signal as a 1D or 2D array.

    Returns:
        RMS value of the spectral signal.
    """
    return float(np.sqrt(np.sum(np.abs(spectrums) ** 2)))

def calculate_spectral_rms_dbm(spectrums: np.ndarray, imped_ohms: float = DEFAULT_IMPED_OHMS) -> float:
    """Calculate the RMS value of a signal in dBm.

    For spectral signals, the RMS is calculated as the square root of the SUM of the squared magnitudes of the spectrums.

    Args:
        spectrums: Input spectral signal (1D or 2D array).
        imped_ohms: Impedance in ohms.

    Returns:
        RMS power in dBm.
    """
    return float( voltage_to_dbm(calculate_spectral_rms(spectrums), imped_ohms) )  # Convert RMS voltage to dBm

def thermal_noise_power_dbm(temp_kelvin: float, bandwidth: float) -> float:
    """Calculate thermal noise power in dBm based on temperature and bandwidth.

    Args:
        temp_kelvin: Temperature in Kelvin.
        bandwidth: Bandwidth in Hertz.

    Returns:
        Thermal noise power in dBm.
    """
    return watts_to_dbm(K_B * temp_kelvin * bandwidth)  # P = K_B * T * B, then convert to dBm

# ====================================================================================================
# Signal Processing Functions
# ====================================================================================================

def compute_spectrums(sigxd: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the frequency spectrum of a signal using FFT.

    Args:
        sigxd: Input signal (1D or 2D array).
        sampling_rate: Sampling rate in Hz.

    Returns:
        Frequency bins and corresponding spectrums.
    """
    if len(sigxd.shape) == 1:
        sig2d = sigxd.reshape(1, sigxd.shape[0])  # Convert 1D to 2D for consistent processing
    else:
        sig2d = sigxd

    n_points  = sig2d.shape[1]                               # Number of samples in each window
    freqs     = np.fft.fftfreq(n_points, 1 / sampling_rate)  # Frequency bins
    spectrums = np.fft.fft(sig2d, axis=1) / n_points         # FFT normalized by number of points

    return freqs, spectrums

def get_spectrums_power_n_phase(freqs: np.ndarray, spectrums: np.ndarray, imped_ohms: float = DEFAULT_IMPED_OHMS) -> tuple[np.ndarray, np.ndarray]:
    """Compute the power and phase spectrum from FFT results.

    Args:
        freqs: Frequency bins.
        spectrums: Complex FFT spectrums (1D or 2D).
        imped_ohms: Impedance in ohms, defaults to 50 ohms.

    Returns:
        Power spectrum in dBm and phase spectrum in radians.
    """
    spects_amp   = np.abs(spectrums)  # Amplitude of the spectrum
    spects_power = voltage_to_dbm(spects_amp, imped_ohms)  # Convert amplitude to power in dBm
    spects_phase = np.angle(spectrums)  # Phase in radians

    return spects_power, spects_phase

# ====================================================================================================
# Visualization Functions
# ====================================================================================================

def plot_temporal_signal(time: np.ndarray, sigxd: np.ndarray, tmin: float | None = None, tmax: float | None = None,
                         title: str = "Temporal Signal", ylabel: str = "Amplitude" ) -> None:
    """Plot the temporal representation of a signal with automatic unit scaling.

    Args:
        time: Time array in seconds.
        sigxd: Signal array (1D or 2D).
        tmin: Minimum time to plot, defaults to signal start.
        tmax: Maximum time to plot, defaults to signal end.
        title: Plot title.
        ylabel: Y-axis label.
    """
    tmin = tmin if tmin is not None else 0.0         # Default to start of signal
    tmax = tmax if tmax is not None else time.max()  # Default to end of signal
    idx_min = np.argmin(np.abs(time - tmin))         # Index of closest time to tmin
    idx_max = np.argmin(np.abs(time - tmax))         # Index of closest time to tmax

    # Determine appropriate time unit for plotting
    if (time[-1] - time[0]) > 10:
        unit = 's'  # Seconds
    elif (time[-1] - time[0]) > 10e-3:
        time = time * 1e3  # Convert to milliseconds
        unit = 'ms'
    elif (time[-1] - time[0]) > 10e-6:
        time = time * 1e6  # Convert to microseconds
        unit = 'us'
    else:
        time = time * 1e9  # Convert to nanoseconds
        unit = 'ns'

    fig, ax = plt.subplots(figsize=(12, 6))  # Create figure and axis
    if len(sigxd.shape) == 1:
        ax.plot(time[idx_min:idx_max], sigxd[idx_min:idx_max])  # Plot 1D signal
    elif len(sigxd.shape) == 2:
        for idx in range(sigxd.shape[0]):
            ax.plot(time[idx_min:idx_max], sigxd[idx, idx_min:idx_max])  # Plot each window of 2D signal

    ax.set_xlabel(f'Time ({unit})')  # Label x-axis with time unit
    ax.set_ylabel(ylabel)  # Set y-axis label
    ax.set_title(title)  # Set plot title
    ax.grid(True)  # Add grid
    plt.tight_layout()  # Adjust layout

def plot_signal_spectrum(freqs: np.ndarray, spectrum_power: np.ndarray, spectrum_phase: np.ndarray | None = None,
                         title_power: str = "Power Spectrum", title_phase: str = "Phase Spectrum",
                         ylabel_power: str = "Power (dBm)", ylabel_phase: str = "Phase (radians)") -> None:
    """Plot the frequency power spectrum and optionally the phase spectrum with automatic unit scaling.

    Args:
        freqs: Frequency bins in Hz.
        spectrum_power: Power spectrum in dBm (1D or 2D).
        spectrum_phase: Phase spectrum in radians (1D or 2D).
        title_power: Title for power spectrum plot.
        title_phase: Title for phase spectrum plot.
        ylabel_power: Y-axis label for power spectrum.
        ylabel_phase: Y-axis label for phase spectrum.
    """
    freqs   = np.array(freqs)
    idx_max = np.argmax(freqs) + 1  # Index of maximum positive frequency

    # Determine appropriate frequency unit for plotting
    if freqs.max() - freqs[0] > 10e9:
        freqs = freqs / 1e9
        unit = 'GHz'
    elif freqs.max() - freqs[0] > 10e6:
        freqs = freqs / 1e6
        unit = 'MHz'
    elif freqs.max() - freqs[0] > 10e3:
        freqs = freqs / 1e3
        unit = 'kHz'
    else:
        unit = 'Hz'

    fig, axes = plt.subplots(2 if spectrum_phase is not None else 1, 1, figsize=(12, 6), sharex=True)
    if spectrum_phase is None:
        axes = [axes]

    if len(spectrum_power.shape) == 1:
        axes[0].plot(freqs[:idx_max], spectrum_power[:idx_max])
        if spectrum_phase is not None:
            axes[1].plot(freqs[:idx_max], spectrum_phase[:idx_max])
    elif len(spectrum_power.shape) == 2:
        for idx in range(spectrum_power.shape[0]):
            axes[0].plot(freqs[:idx_max], spectrum_power[idx, :idx_max])
            if spectrum_phase is not None:
                axes[1].plot(freqs[:idx_max], spectrum_phase[idx, :idx_max])

    ymin, ymax = max(spectrum_power.min(), -1000.), min(spectrum_power.max(), 1000.)
    axes[0].set_ylim(ymin, ymax)
    axes[0].set_xlabel(f'Frequency ({unit})')
    axes[0].set_ylabel(ylabel_power)
    axes[0].set_title(title_power)
    axes[0].grid(True)

    if spectrum_phase is not None:
        axes[1].set_ylim(-pi, pi)
        axes[1].set_xlabel(f'Frequency ({unit})')
        axes[1].set_ylabel(ylabel_phase)
        axes[1].set_title(title_phase)
        axes[1].grid(True)

    plt.tight_layout()

# ====================================================================================================
# Signals Class
# ====================================================================================================

class Signals:
    """Class for managing and processing multiple microwave signals.

    Attributes:
        fmax (float): Maximum frequency of signals in Hz.
        bin_width (float): Width of the spectral bins in Hz.
        n_windows (int): Number of signal windows.
        imped_ohms (float): Impedance in ohms.
        temp_kelvin (float): Temperature in Kelvin.
        duration (float): Duration of each signal window in seconds.
        n_points (int): Number of samples per window.
        sampling_rate (float): Sampling rate in Hz.
        freqs (np.ndarray): Frequency bins for FFT.
        bandwidth (float): Nyquist Bandwidth in Hz (half of sampling rate).
        shape (Tuple[int, int]): Shape of the signal array (n_windows, n_points).
        time (np.ndarray): Time array for each window.
        sig2d (np.ndarray): 2D array of signals (windows x samples).
        _spectrum_uptodate (bool): Flag indicating if spectrum is up-to-date.
        _spectrums (Optional[np.ndarray]): Cached FFT spectrums.
        _spects_power (Optional[np.ndarray]): Cached power spectrums in dBm.
        _spects_phase (Optional[np.ndarray]): Cached phase spectrums in radians.
    """

    def __init__(self, fmax: float, bin_width: float, n_windows: int = DEFAULT_N_WINDOWS,
                 imped_ohms: float = DEFAULT_IMPED_OHMS, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        """Initialize the Signals object with specified parameters.

        Args:
            fmax: Maximum frequency of signals in Hz.
            bin_width: Width of the spectral bins in Hz.
            n_windows: Number of signal windows, defaults to 32.
            imped_ohms: Impedance in ohms, defaults to 50 ohms.
            temp_kelvin: Temperature in Kelvin, defaults to 298.15 K.
        """
        self.fmax        = fmax
        self.bin_width   = bin_width
        self.imped_ohms  = imped_ohms
        self.temp_kelvin = temp_kelvin
        self.n_windows   = n_windows

        self.duration      = 1 / bin_width  # Duration is inverse of bin width
        self.n_points      = int(np.ceil(2.2 * fmax / bin_width))  # Ensure sufficient sampling
        self.sampling_rate = self.n_points / self.duration  # Sampling rate in Hz
        self.freqs         = np.fft.fftfreq(self.n_points, 1 / self.sampling_rate)  # Frequency bins

        self.bandwidth = self.sampling_rate / 2  # Nyquist bandwidth

        logger.debug(f'<{self.__class__.__name__}> fmax={self.fmax/1e9:.3f} GHz, bin_width={self.bin_width/1e6:.3f} MHz, bandwidth={self.bandwidth/1e6:.3f} MHz, sampling_rate={self.sampling_rate/1e9:.3f} GHz, n_points={self.n_points}, duration={self.duration*1e6:.3f} µs')

        self.shape = (self.n_windows, self.n_points)  # Shape of signal array

        self.time = np.linspace(0, self.duration, self.n_points, endpoint=False)  # Time array
        self.sig2d = Signals.generate_noise_dbm(self.shape, -1000)  # Initialize with very low noise

        self._spectrum_uptodate = False
        self._spectrums         = None
        self._spects_power      = None
        self._spects_phase      = None

    @staticmethod
    def generate_signal_dbm(time: np.ndarray, freq: float, power_dbm: float, phase: float,
                            imped_ohms: float = DEFAULT_IMPED_OHMS) -> np.ndarray:
        """Generate a monotone sinusoidal signal with specified power and phase.

        Args:
            time: Time array in seconds.
            freq: Frequency in Hz.
            power_dbm: Power in dBm.
            phase: Phase in radians.
            imped_ohms: Impedance in ohms.

        Returns:
            Generated signal array.
        """
        amp_rms = dbm_to_voltage(power_dbm, imped_ohms)  # RMS amplitude
        if freq == 0:
            signal = np.full_like(time, amp_rms)  # DC signal with constant amplitude
        else:
            signal  = np.sin(2 * pi * freq * time + phase)     # Sine wave with given frequency and phase
            factor  = amp_rms / np.sqrt(np.mean(signal ** 2))  # Scaling factor to achieve desired RMS
            signal *= factor

        return signal

    @staticmethod
    def generate_noise_dbm( shape: int | tuple[int, ...], power_dbm: float, imped_ohms: float = DEFAULT_IMPED_OHMS) -> np.ndarray:
        """Generate Gaussian noise with specified power in dBm.

        Args:
            shape: Shape of the noise array.
            power_dbm: Power in dBm.
            imped_ohms: Impedance in ohms.

        Returns:
            Generated noise array.
        """
        amp = dbm_to_voltage(power_dbm, imped_ohms)  # amplitude
        return np.random.normal(0, amp, shape)  # Gaussian noise with zero mean

    def compute_spectrum(self, force: bool = False) -> None:
        """Compute the frequency spectrum of the signals if not up-to-date.

        Args:
            force: Force recomputation even if spectrum is up-to-date.
        """
        if force or not self._spectrum_uptodate:
            self.freqs, self._spectrums = compute_spectrums(self.sig2d, self.sampling_rate)
            self._spects_power, self._spects_phase = get_spectrums_power_n_phase(self.freqs, self._spectrums, self.imped_ohms)
            self._spectrum_uptodate = True

    @property
    def spectrums(self) -> np.ndarray:
        """Get the complex FFT spectrums of the signals.

        Returns:
            np.ndarray: Complex spectrums (n_windows x n_points).
        """
        self.compute_spectrum()
        return self._spectrums

    @property
    def spects_power(self) -> np.ndarray:
        """Get the power spectrums of the signals in dBm.

        Returns:
            np.ndarray: Power spectrums (n_windows x n_points).
        """
        self.compute_spectrum()
        return self._spects_power

    @property
    def spects_phase(self) -> np.ndarray:
        """Get the phase spectrums of the signals in radians.

        Returns:
            np.ndarray: Phase spectrums (n_windows x n_points).
        """
        self.compute_spectrum()
        return self._spects_phase

    def get_arg_freq(self, freq: float) -> int:
        """Get the index of the closest frequency in the spectrum.

        Args:
            freq: Target frequency in Hz.

        Returns:
            Index of the closest frequency bin.
        """
        return int(np.argmin(np.abs(self.freqs - freq)))

    def add_signal(self, sigxd: np.ndarray) -> None:
        """Add a signal to the existing signals.

        Args:
            sigxd: Signal to add (1D or 2D array).

        Raises:
            ValueError: If the shape of sigxd is incompatible.
        """
        if sigxd.shape == self.sig2d.shape:
            self.sig2d += sigxd
        elif len(sigxd.shape) == 2 and sigxd.shape[0] == 1 and sigxd.shape[1] == self.sig2d.shape[1]:
            self.sig2d[:] += sigxd[0]
        elif len(sigxd.shape) == 1 and sigxd.shape[0] == self.sig2d.shape[1]:
            self.sig2d[:] += sigxd
        else:
            raise ValueError(f"Invalid input, sigxd.shape: {sigxd.shape}")

        self._spectrum_uptodate = False

    def add_tone(self, freq: float, power_dbm: float, phase: float) -> None:
        """Add a sinusoidal tone to the signals.

        Args:
            freq: Frequency in Hz.
            power_dbm: Power in dBm.
            phase: Phase in radians.
        """
        tone = self.generate_signal_dbm(self.time, freq, power_dbm, phase, self.imped_ohms)
        self.sig2d += tone
        self._spectrum_uptodate = False

    def add_noise(self, power_dbm: float) -> None:
        """Add Gaussian noise to the signals.

        Args:
            power_dbm: Power in dBm.
        """
        noise = self.generate_noise_dbm(self.shape, power_dbm, self.imped_ohms)
        self.sig2d += noise
        self._spectrum_uptodate = False

    def add_thermal_noise(self, temp_kelvin: float | None = None) -> None:
        """Add thermal noise to the signals based on temperature and bandwidth.

        Args:
            temp_kelvin: Temperature in Kelvin.
        """
        temp_kelvin = temp_kelvin if temp_kelvin is not None else self.temp_kelvin
        noise_power_dbm = thermal_noise_power_dbm(temp_kelvin, self.bandwidth)
        self.add_noise(noise_power_dbm)

    def rms_dbm(self) -> float:
        """Calculate the RMS value of the signals in dBm.

        Returns:
            RMS power in dBm.
        """
        return voltage_to_dbm(np.sqrt(np.mean(np.abs(self.sig2d) ** 2)), self.imped_ohms)

    def rms_at_freq(self, freq: float) -> float:
        """Calculate the exact signal RMS voltage at a given frequency.

        This uses time-domain convolution (Single-point DTFT / Heterodyning) to avoid 
        spectral leakage issues and FFT grid constraints by correlating the signal 
        directly with a complex phasor at the exact frequency.

        Args:
            freq (float): The frequency to analyze in Hz.

        Returns:
            float: The exact RMS voltage at that frequency.
        """
        phasor      = np.exp(-1j * 2 * pi * freq * self.time)

        #window      = np.hanning(self.n_points)
        #complex_amp = np.mean(self.sig2d * phasor * window) / np.mean(window)

        complex_amp = np.mean(self.sig2d * phasor)

        if freq < self.bin_width:
            phasor         = np.exp(-1j * 2 * pi * (-freq) * self.time)
            complex_amp   += np.conj(np.mean(self.sig2d * phasor))  # Add negative frequency component for real signals
            rms_per_window = np.abs(complex_amp)/2
        else:
            rms_per_window = np.sqrt(2) * np.abs(complex_amp)

        return float(np.sqrt(np.mean(rms_per_window**2)))

    def rms_at_freq_dbm(self, freq: float) -> float:
        """Calculate the exact signal RMS power in dBm at a given frequency.

        Args:
            freq (float): The frequency to analyze in Hz.

        Returns:
            float: The total power in dBm at that frequency.
        """
        rms_voltage = self.rms_at_freq(freq)
        return float(voltage_to_dbm(rms_voltage, self.imped_ohms))

    def plot_temporal(self, tmin: float | None = None, tmax: float | None = None, title: str = "Temporal Signal", ylabel: str = "Amplitude") -> None:
        """Plot the temporal representation of the signals.

        Args:
            tmin: Minimum time to plot.
            tmax: Maximum time to plot.
            title: Plot title.
            ylabel: Y-axis label.
        """
        plot_temporal_signal(self.time, self.sig2d, tmin, tmax, title, ylabel)

    def plot_spectrum(self, title_power: str = "Power Spectrum", title_phase: str = "Phase Spectrum",
                      ylabel_power: str = "Power (dBm)", ylabel_phase: str = "Phase (radians)") -> None:
        """Plot the frequency spectrum of the signals (power and phase).

        Args:
            title_power: Title for power spectrum plot.
            title_phase: Title for phase spectrum plot.
            ylabel_power: Y-axis label for power spectrum.
            ylabel_phase: Y-axis label for phase spectrum.
        """
        exp3      = min(3, int(np.log10(self.bin_width)/3.))
        bin_width = self.bin_width / (10**(3*exp3))   # Scale bin width for unit representation
        unit      = ["Hz", "kHz", "MHz", "GHz"][exp3] # Determine unit

        title_power += f" with Resolution Bandwidth (bin width): {bin_width:.1f} {unit}"
        plot_signal_spectrum(self.freqs, self.spects_power, self.spects_phase, title_power, title_phase, ylabel_power, ylabel_phase)

# ====================================================================================================
# RF Component Base Class
# ====================================================================================================

class RF_Abstract_Base_Component(ABC):
    """Abstract base class for RF components.

    Provides common functionality and interface for all RF components.
    """

    @staticmethod
    def ft(x: np.ndarray, k: float) -> np.ndarray:
        """Apply a hyperbolic tangent function scaled by k to limit signal amplitude.

        Args:
            x: Input signal array.
            k: Scaling factor for amplitude limiting.

        Returns:
            Transformed signal array.
        """
        return k * np.tanh(x / k)

    @abstractmethod
    def process_signals(self, signals: Signals, temp_kelvin: float | None = None) -> None:
        """Abstract method to process signals. Must be implemented by subclasses.

        Args:
            signals: Input signals object to process.
            temp_kelvin: Temperature in Kelvin.
        """
        pass

    def process(self, signals: Signals, temp_kelvin: float | None = None, inplace: bool = True) -> Signals | None:
        """Process signals through the RF component, optionally returning a new object.

        Args:
            signals: Input signals object.
            temp_kelvin: Temperature in Kelvin.
            inplace: If True, modify signals in place; if False, return a new Signals object.

        Returns:
            Processed signals if inplace is False, otherwise None.
        """
        if not inplace:
            signals = copy.deepcopy(signals)

        means = (1.0 - 1e-3) * signals.sig2d.mean(axis=1, keepdims=True)  # Calculate DC offset
        signals.sig2d -= means                                            # DC block: remove continuous signal

        self.process_signals(signals, temp_kelvin=temp_kelvin)
        signals._spectrum_uptodate = False

        return signals if not inplace else None

    def assess_gain(self, fmin: float = 400e6, fmax: float = 19e9, fstp: float = 100e6, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Assess the gain, phase, and noise figure versus frequency of the RF component.

        Args:
            fmin: Minimum frequency in Hz.
            fmax: Maximum frequency in Hz.
            fstp: Frequency step in Hz.
            temp_kelvin: Temperature in Kelvin.

        Returns:
            Tuple of Frequencies, gains (dB), phases (radians), noise figures (dB).
        """
        logger.info( f"<{self.__class__.__name__}> Assess the gain and phase versus frequency of the RF component.")

        bin_width = fstp / 2
        n_windows = 128

        # Initialize noisy signals
        noisy_signals = Signals(fmax, bin_width, n_windows=n_windows, imped_ohms=50, temp_kelvin=temp_kelvin)
        noisy_signals.add_thermal_noise(temp_kelvin=temp_kelvin)

        # Frequencies for testing
        freqs = np.arange(fmin, fmax+0.1*fstp, fstp) # fmax+0.1*fstp: in order to include fmax

        gains = np.zeros_like(freqs)
        phass = np.zeros_like(freqs)
        n_fgs = np.zeros_like(freqs)

        for idx_frq, freq in tqdm(enumerate(freqs), total=len(freqs), desc="Assessing Gain"):
            # Compute gains and phases
            clear_signal = Signals(fmax, bin_width, n_windows=1, imped_ohms=50, temp_kelvin=temp_kelvin)
            clear_signal.add_tone(freq, -50, 0)

            arg_frq_pos = clear_signal.get_arg_freq(freq)
            arg_frq_neg = clear_signal.get_arg_freq(-freq)

            gains[idx_frq] = clear_signal.spects_power[0, arg_frq_pos]
            phass[idx_frq] = clear_signal.spects_phase[0, arg_frq_pos]

            proc_clear_signal = self.process(clear_signal, temp_kelvin=temp_kelvin, inplace=False)

            gains[idx_frq] = proc_clear_signal.spects_power[0, arg_frq_pos] - gains[idx_frq]
            phass[idx_frq] = proc_clear_signal.spects_phase[0, arg_frq_pos] - phass[idx_frq]

            # Compute noise figure
            signals = copy.deepcopy(noisy_signals)
            signals.add_signal(clear_signal.sig2d)

            spects_befor = (signals.spectrums[:, arg_frq_neg] +
                            np.conj(signals.spectrums[:, arg_frq_pos])) / 2

            self.process(signals, temp_kelvin=temp_kelvin)

            spects_after = (signals.spectrums[:, arg_frq_neg] +
                            np.conj(signals.spectrums[:, arg_frq_pos])) / 2

            pwr_sig_befor = voltage_to_dbm(np.abs(spects_befor.mean()))
            pwr_sig_after = voltage_to_dbm(np.abs(spects_after.mean()))

            pwr_nse_befor = voltage_to_dbm(np.sqrt(spects_befor.var()))
            pwr_nse_after = voltage_to_dbm(np.sqrt(spects_after.var()))

            n_fgs[idx_frq] = (pwr_sig_befor - pwr_nse_befor) - (pwr_sig_after - pwr_nse_after)

        phass[phass < -pi] += 2 * pi  # Wrap phases to [-π, π]
        phass[phass >  pi] -= 2 * pi

        #for var in ('freqs', 'gains', 'phass, n_fgs'):
            #logger.info( f'<{self.__class__.__name__}> {var}: {eval(var)}' )

        return freqs, gains, phass, n_fgs

    def assess_ipx_for_freq(self, fc: float = 9e9, df: float = 100e6, temp_kelvin: float = DEFAULT_TEMP_KELVIN, toplot: bool = False) -> tuple[float, float, float, float, float, float]:
        """Assess the IP2 and IP3 (Intercept Points of order 2 and 3) for a specific frequency.

        Args:
            fc: Center frequency in Hz.
            df: Frequency separation in Hz.
            temp_kelvin: Temperature in Kelvin.
            toplot: Generate plots if True.

        Returns:
            Tuple of (Gain dB, OP1dB dBm, IIP3 dBm, OIP3 dBm, IIP2 dBm, OIP2 dBm).
        """
        logger.info( f'<{self.__class__.__name__}> ## Assess the IP2 and IP3 (Intercept Point of order 2 and 3) of the RF component.' )

        # --------------------------------------------------------
        fmax      = 3. * fc
        bin_width = df / 4 #32
        n_windows = 1
        # --------------------------------------------------------

        # --------------------------------------------------------
        input_pwr_m, outpt_pwr_m = [], []
        input_pwr_d, outpt_pwr_d, im2___power, im3___power = [], [], [], []

        gains_db = []
        ip1db_dbm, op1db_dbm = None, None
        # --------------------------------------------------------

        # --------------------------------------------------------
        f1 = fc - df
        f2 = fc + df
        f1pf2 = f1 + f2
        df1mf2 = 2 * f1 - f2
        df2mf1 = 2 * f2 - f1
        logger.info( f'<{self.__class__.__name__}>    Central frequency: {fc / 1e9:.3f} GHz, df: {df / 1e6:.3f} MHz, F1: {f1 / 1e9:.3f} GHz, F2: {f2 / 1e9:.3f} GHz' )

        # Indexes of frequencies in spectrum
        signals = Signals(fmax, bin_width, n_windows=1, imped_ohms=50, temp_kelvin=temp_kelvin)

        arg_frq_fc = signals.get_arg_freq(fc)
        arg_frq_f1 = signals.get_arg_freq(f1)
        arg_frq_f2 = signals.get_arg_freq(f2)
        arg_frq_f1pf2 = signals.get_arg_freq(f1pf2)
        arg_frq_df1mf2 = signals.get_arg_freq(df1mf2)
        arg_frq_df2mf1 = signals.get_arg_freq(df2mf1)
        # --------------------------------------------------------

        # --------------------------------------------------------
        for power_dbm in tqdm(range(-50, 50+1), desc="Assessing Gain"):
            # Initialize monotone signals
            signals = Signals(fmax, bin_width, n_windows=n_windows, imped_ohms=50, temp_kelvin=temp_kelvin)
            signals.add_tone(fc, power_dbm, 0)

            input_pwr_m.append( np.mean(signals.spects_power[:, arg_frq_fc]) )

            # Processing signal
            self.process(signals, temp_kelvin=temp_kelvin)

            # Append values in tables
            outpt_pwr_m.append( np.mean(signals.spects_power[:, arg_frq_fc]) )

            if len(input_pwr_m) >= 2:
                delta_input_power = input_pwr_m[-1] - input_pwr_m[-2]
                delta_outpt_power = outpt_pwr_m[-1] - outpt_pwr_m[-2]

                if op1db_dbm is None:
                    if np.abs(delta_outpt_power / 1 - delta_input_power) / delta_input_power < 0.02:
                        gains_db.append( outpt_pwr_m[-1] - input_pwr_m[-1] )
                        gain_db = np.array(gains_db).mean()

                    if gains_db and (input_pwr_m[-1] + gain_db - outpt_pwr_m[-1]) > 1:
                        ip1db_dbm = np.interp(1.,
                                              [input_pwr_m[-2]+gain_db-outpt_pwr_m[-2], input_pwr_m[-1]+gain_db-outpt_pwr_m[-1]],
                                              [input_pwr_m[-2]                        , input_pwr_m[-1]                        ],
                                              )

                        op1db_dbm = np.interp(ip1db_dbm,
                                              [input_pwr_m[-2], input_pwr_m[-1]],
                                              [outpt_pwr_m[-2], outpt_pwr_m[-1]],
                                              )

        if not gains_db:
            logger.info( f'<{self.__class__.__name__}> Error: Unable to characterize Gain finely.' )
            gain_db = outpt_pwr_m[len(outpt_pwr_m)//2] - input_pwr_m[len(outpt_pwr_m)//2]

        if op1db_dbm is None:
            logger.info( f'<{self.__class__.__name__}> Error: Unable to characterize P1dB finely.' )
            ip1db_dbm = input_pwr_m[-1]
            op1db_dbm = outpt_pwr_m[-1]

        input_pwr_m = np.array(input_pwr_m)
        outpt_pwr_m = np.array(outpt_pwr_m)
        h1_slope, h1_inter = search_mediane_for_slope(input_pwr_m, outpt_pwr_m, 1)
        logger.info( f'<{self.__class__.__name__}> h1_slope: {h1_slope}, h1_inter: {h1_inter}' )
        # --------------------------------------------------------

        # --------------------------------------------------------
        for power_dbm in tqdm(range(-50, 50+1), desc="Assessing IP2, IP3"):
            # Initialize bitone signals
            signals = Signals(fmax, bin_width, n_windows=n_windows, imped_ohms=50, temp_kelvin=temp_kelvin)
            signals.add_tone(f1, power_dbm, 0)
            signals.add_tone(f2, power_dbm, 0)

            input_pwr_d.append(np.mean((signals.spects_power[:, arg_frq_f1] +
                                        signals.spects_power[:, arg_frq_f2]) / 2))

            # Processing signal
            self.process(signals, temp_kelvin=temp_kelvin)

            # Append values in tables
            outpt_pwr_d.append(np.mean((signals.spects_power[:, arg_frq_f1] +
                                        signals.spects_power[:, arg_frq_f2]) / 2))

            im2___power.append(np.mean(signals.spects_power[:, arg_frq_f1pf2]))

            im3___power.append(np.mean((signals.spects_power[:, arg_frq_df1mf2] +
                                        signals.spects_power[:, arg_frq_df2mf1]) / 2))

        input_pwr_d = np.array(input_pwr_d)
        im3___power = np.array(im3___power)
        im2___power = np.array(im2___power)

        # Retrieving IP3
        im3_slope, im3_inter = search_mediane_for_slope(input_pwr_d, im3___power, 3)
        if np.count_nonzero((np.abs(im3___power - (im3_slope * input_pwr_d + im3_inter)) < 1.)) / len(input_pwr_d) > 5e-2:
            iip3_dbm = (gain_db - im3_inter) / 2
        else:
            logger.info("Error: Unable to characterize IP3 finely.")
            iip3_dbm = op1db_dbm + 14 - gain_db

        oip3_dbm = iip3_dbm + gain_db

        # Retrieving IP2
        im2_slope, im2_inter = search_mediane_for_slope(input_pwr_d, im2___power, 2)
        if np.count_nonzero((np.abs(im2___power - (im2_slope * input_pwr_d + im2_inter)) < 1.)) / len(input_pwr_d) > 5e-2:
            iip2_dbm = gain_db - im2_inter
        else:
            logger.info( f'<{self.__class__.__name__}> Error: Unable to characterize IP2 finely.' )
            iip2_dbm = iip3_dbm + 25

        oip2_dbm = iip2_dbm + gain_db if iip2_dbm is not None else None
        # --------------------------------------------------------

        # --------------------------------------------------------
        # Plotting
        if toplot:
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.subplots(1, 1)
            ax1.axis('equal')

            ax1.plot( ip1db_dbm, op1db_dbm, 'go' )
            ax1.plot( iip2_dbm , oip2_dbm , 'mo' )
            ax1.plot( iip3_dbm , oip3_dbm , 'ro' )

            ax1.plot( [input_pwr_m[0], input_pwr_m[-1]], [   input_pwr_m[0]+gain_db              , input_pwr_m[-1]+gain_db                 ], 'g:' )
            ax1.plot( [input_pwr_d[0], input_pwr_d[-1]], [3*(input_pwr_d[0]+gain_db) - 2*oip3_dbm, 3*(input_pwr_d[-1]+gain_db) - 2*oip3_dbm], 'r:' )
            ax1.plot( [input_pwr_d[0], input_pwr_d[-1]], [2*(input_pwr_d[0]+gain_db) - 1*oip2_dbm, 2*(input_pwr_d[-1]+gain_db) - 1*oip2_dbm], 'm:' )

            for in_pwrs, out_pwrs, line in ((input_pwr_m, outpt_pwr_m, 'g'), (input_pwr_d, im2___power, 'm'), (input_pwr_d, im3___power, 'r')):
                ax1.plot(in_pwrs, out_pwrs, line)

            ax1.set_xlabel('Input power (dBm)')
            ax1.set_ylabel('Output powers (dBm)')
            ax1.set_xticks(np.arange(-100, 100+1, 10))
            ax1.set_yticks(np.arange(-100, 100+1, 10))
            ax1.set_ylim(-80, 70)
            ax1.grid(True)
            plt.tight_layout()

            for var_name in ('gain_db', 'op1db_dbm', 'iip3_dbm', 'oip3_dbm', 'iip2_dbm', 'oip2_dbm'):
                logger.info( f'<{self.__class__.__name__}> {var_name}: {eval(var_name)}' )
        # --------------------------------------------------------

        return gain_db, op1db_dbm, iip3_dbm, oip3_dbm, iip2_dbm, oip2_dbm

    def assess_ipx(self, fmin: float, fmax: float, fstp: float = 1e9, df: float = 200e6, temp_kelvin: float = DEFAULT_TEMP_KELVIN, toplot: bool = False) -> np.ndarray:
        """Assess the IP2 and IP3 for a range of frequencies.

        Args:
            fmin: Minimum frequency in Hz.
            fmax: Maximum frequency in Hz.
            fstp: Frequency step in Hz.
            df: Frequency separation.
            temp_kelvin: Temperature in Kelvin.
            toplot: Generate plot if True.

        Returns:
            Numpy array of assessed metrics across frequencies (frequency, (gain_db, op1db_dbm, iip2_dbm, oip3_dbm)).
        """
        results = []
        freqs = np.arange(fmin, fmax+0.1*fstp, fstp) # fmax+0.1*fstp: in order to include fmax
        for fc in freqs:
            result = self.assess_ipx_for_freq(fc, df=df, temp_kelvin=temp_kelvin)  # df now variable
            results.append([fc] + list(result))

        results = np.array(results).T

        if toplot:
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.subplots(1, 1)

            for pwr, line, label in ((results[1], 'g', 'gain_db' ), (results[2], 'g:', 'op1db_dbm'),
                                     (results[3], 'r', 'iip3_dbm'), (results[4], 'r:', 'oip3_dbm' ),
                                     (results[5], 'm', 'iip2_dbm'), (results[6], 'm:', 'oip2_dbm')):
                ax1.plot(freqs/1e9, pwr, line, label=label)

            ax1.legend()
            ax1.set_xlabel('Frequency (GHz)')
            ax1.set_ylabel('Power (dB, dBm)')
            ax1.grid(True)
            plt.tight_layout()

        return results

# ====================================================================================================
# RF Channel Class
# ====================================================================================================

class RF_chain(RF_Abstract_Base_Component):
    """Class representing an RF chain composed of multiple RF components.

    Attributes:
        rf_components (list): List of RF_Abstract_Base_Component instances in the chain.
    """

    def __init__(self, rf_components: list[RF_Abstract_Base_Component]) -> None:
        """Initialize the RF chain with a list of components.

        Args:
            rf_components: List of RF_Abstract_Base_Component objects.
        """
        self.rf_components = list(rf_components)

    def process_signals(self, signals: Signals, temp_kelvin: float | None = None) -> None:
        """Process signals through each RF component in the chain sequentially.

        Args:
            signals: Input signals object.
            temp_kelvin: Temperature in Kelvin.
        """
        for rf_compnt in self.rf_components:
            rf_compnt.process(signals, temp_kelvin=temp_kelvin)

# ====================================================================================================
# RF Modelised Component Class
# ====================================================================================================

switch_print = False
class RF_Abstract_Modelised_Component(RF_Abstract_Base_Component, ABC):
    """Abstract base class representing a modelised RF component with frequency-dependent characteristics.

    Provides common functionality and interface for all RF modelised components.
    """
    # Define iip3 coefficient to use after signal normalisation (s/iip3_equiv_gain)
    k_op1  =  2.2
    k_iip3 =  9.2e-1
    k_iip2 = -0.5
    k_iip2_sat = 1.5

    # Threshold : for each op1_S, iip3_s, iip_2s if any value is below threshold the effect is processed 
    iipx_threshold = dbm_to_voltage(1000)

    # Define out-of-band frequency and gain characteristics
    freqs_sup    = 100e6 * np.arange(1, 11) # 100 MHz steps above band
    freqs_inf    = -freqs_sup[::-1]         # Mirror below band

    gains_sup_db = -5.0 * np.arange(1, 11)  # -5 dB per step
    gains_inf_db = gains_sup_db[::-1]       # -5 dB per step

    gains_sup = gain_db_to_gain(gains_sup_db)
    gains_inf = gain_db_to_gain(gains_inf_db)

    nfs___sup = nf_db_to_nf(-gains_sup_db)
    nfs___inf = nf_db_to_nf(-gains_inf_db)

    @abstractmethod
    def get_rf_parameters_adapted_to_signals(self, signals: Signals, temp_kelvin: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Abstract method to return frequency-dependent gains, noise, op1db, iip3 and iip2 figures for the signals.

        Args:
            signals (Signals): Input signals object.
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to instance temperature.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Frequencies, gains, noise figures, op1db, iip3, iip2.

        Raises:
            ValueError: If frequency-dependent parameters are not set.
        """
        temp_kelvin = temp_kelvin if temp_kelvin is not None else self.temp_kelvin

        return self.extend_rf_parameters(signals, temp_kelvin=temp_kelvin)

    def extend_rf_parameters(self, freqs: np.ndarray, gains: np.ndarray, nf__s: np.ndarray,
                             op1ds: np.ndarray, iip3s: np.ndarray, iip2s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extend frequency-dependent parameters to include out-of-band behavior."""
        # Extend frequency range for out-of-band behavior
        freqs_out = np.concatenate((self.freqs_inf + freqs[0], freqs, self.freqs_sup + freqs[-1]))

        _gains_inf = self.gains_inf * gains[0]
        _gains_sup = self.gains_sup * gains[-1]

        _nfs_inf   = mul_nfs(self.nfs___inf, nf__s[ 0])
        _nfs_sup   = mul_nfs(self.nfs___sup, nf__s[-1])

        gains = np.concatenate((_gains_inf, gains, _gains_sup))
        nf__s = np.concatenate((  _nfs_inf, nf__s  ,   _nfs_sup))

        # Remove negative frequencies
        freq_args = np.flatnonzero(freqs_out >= 0.)
        freqs_out = freqs_out[freq_args]
        gains     = gains[freq_args]
        nf__s     = nf__s[freq_args]

        # Interpolate op1dB, iip3, and iip2
        op1ds = np.interp(freqs_out, freqs, op1ds, left=op1ds[0], right=op1ds[-1])
        iip3s = np.interp(freqs_out, freqs, iip3s, left=iip3s[0], right=iip3s[-1])
        iip2s = np.interp(freqs_out, freqs, iip2s, left=iip2s[0], right=iip2s[-1])

        return freqs_out, gains, nf__s, op1ds, iip3s, iip2s

    def process_signals(self, signals: Signals, temp_kelvin: float | None = None) -> None:
        """Process signals by applying frequency-dependent gains, noise, and distortion.

        Args:
            signals: Input signals object.
            temp_kelvin: Temperature in Kelvin.
        """
        temp_kelvin = temp_kelvin if temp_kelvin is not None else self.temp_kelvin

        # Get RF parameter figures
        freqs, gains, nf__s, op1ds, iip3s, iip2s = self.get_rf_parameters_adapted_to_signals(signals, temp_kelvin)

        # Extend to negative frequencies (symmetric response)
        if np.count_nonzero(freqs<0.) == 0:
            gains = np.concatenate((np.conjugate(gains[freqs > 0][::-1]), gains[freqs >= 0]))
            nf__s = np.concatenate((nf__s[freqs > 0][::-1], nf__s[freqs >= 0]))

            op1ds = np.concatenate((op1ds[freqs > 0][::-1], op1ds[freqs >= 0]))
            iip3s = np.concatenate((iip3s[freqs > 0][::-1], iip3s[freqs >= 0]))
            iip2s = np.concatenate((iip2s[freqs > 0][::-1], iip2s[freqs >= 0]))

            freqs = np.concatenate((-freqs[freqs > 0][::-1], freqs[freqs >= 0]))

        if np.count_nonzero(freqs<0.) != np.count_nonzero(freqs>0.):
            logger.info( f'<{self.__class__.__name__}> np.count_nonzero(freqs<0.), np.count_nonzero(freqs>0.): {np.count_nonzero(freqs<0.), np.count_nonzero(freqs>0.)}' )
            logger.info( f'<{self.__class__.__name__}> freqs[freqs<0.], freqs[freqs>0.]: {freqs[freqs<0.], freqs[freqs>0.]}' )
            raise ValueError("Frequency-dependent parameters are not set correctly.")

        #for var_name in ('freqs', 'gains', 'nf__s', 'op1ds', 'iip3s', 'iip2s'):
            #logger.info( f'<{self.__class__.__name__}> {var_name}: {eval(var_name)}' )

        # Get spectrums of the signals
        fftfreqs  = np.fft.fftfreq(signals.n_points, 1 / signals.sampling_rate)
        spectrums = np.fft.fft(signals.sig2d, axis=1) / len(fftfreqs)

        # Interpolate gains and noise figures
        gains = np.interp(fftfreqs, freqs, gains, left=gains[0], right=gains[-1])
        nf__s = np.interp(fftfreqs, freqs, nf__s, left=nf__s[0], right=nf__s[-1])
        op1ds = np.interp(fftfreqs, freqs, op1ds, left=op1ds[0], right=op1ds[-1])
        iip3s = np.interp(fftfreqs, freqs, iip3s, left=iip3s[0], right=iip3s[-1])
        iip2s = np.interp(fftfreqs, freqs, iip2s, left=iip2s[0], right=iip2s[-1])

        global switch_print
        if switch_print:
            switch_print = False
            sorted_args = fftfreqs.argsort() 
            plt.figure(figsize=(12, 6))            
            plt.plot(fftfreqs[sorted_args]/1e9, gain_to_gain_db(gains[sorted_args]), 'g-', label='gain')
            plt.plot(fftfreqs[sorted_args]/1e9, nf_to_nf_db(nf__s[sorted_args]), 'k:', label='nf')
            plt.plot(fftfreqs[sorted_args]/1e9, voltage_to_dbm(op1ds[sorted_args]), 'g:', label='op1dB')
            plt.plot(fftfreqs[sorted_args]/1e9, voltage_to_dbm(iip3s[sorted_args]), 'r-', label='iip3')
            plt.plot(fftfreqs[sorted_args]/1e9, voltage_to_dbm(iip3s[sorted_args])+gain_to_gain_db(gains[sorted_args]), 'r:', label='oip3')
            plt.plot(fftfreqs[sorted_args]/1e9, voltage_to_dbm(iip2s[sorted_args]), 'm-', label='iip2')
            plt.plot(fftfreqs[sorted_args]/1e9, voltage_to_dbm(iip2s[sorted_args])+gain_to_gain_db(gains[sorted_args]), 'm:', label='oip2')
            plt.grid()
            plt.legend()

        # Apply noise figures in frequency domain
        noise = Signals.generate_noise_dbm(signals.shape, thermal_noise_power_dbm(temp_kelvin, signals.bandwidth), signals.imped_ohms)
        spectrums += nf__s * np.fft.fft(noise, axis=1) / len(fftfreqs)

        # IIP3 processing
        # Apply third-order non-linearity
        if iip3s.min() < self.iipx_threshold:
            s13 = np.real(np.fft.ifft(spectrums * len(fftfreqs) / iip3s, axis=1))
            s13 = self.ft(s13, self.k_iip3)

            # Retrieve spectrum and apply gain
            spect_13 = gains * iip3s * np.fft.fft(s13, axis=1) / len(fftfreqs)
        else:
            # Apply gain
            spect_13 = spectrums * gains

        # IIP2 processing
        # Apply second-order non-linearity and remove DC component
        if iip2s.min() < self.iipx_threshold:
            s_2  = np.real(np.fft.ifft(spectrums * len(fftfreqs) / iip2s, axis=1))
            s_2  = self.k_iip2 * s_2**2

            # Remove DC component
            s_2 -= s_2.mean(1)[:, np.newaxis]

            # Retrieve spectrum and apply gain
            spect__2 = gains * iip2s * np.fft.fft(s_2, axis=1) / len(fftfreqs)

            # Compression
            if op1ds.min() < self.iipx_threshold:
                spect__2 = self.ft(np.abs(spect__2), op1ds * self.k_iip2_sat) * np.exp(1j * np.angle(spect__2))
        else:
            # No effect
            spect__2 = np.zeros_like(spectrums)

        # Combine effects IPx
        spectrums = spect_13 + spect__2

        # Apply final compression limiting
        if op1ds.min() < self.iipx_threshold:
            spectrums = self.ft(np.abs(spectrums), op1ds * self.k_op1) * np.exp(1j * np.angle(spectrums))

        #logger.info( f'<{self.__class__.__name__}> out: {voltage_to_dbm(np.abs(spectrums[:, idx_frtest]))} dBm' )

        # Retrieve temporal signal
        signals.sig2d = np.real(np.fft.ifft( spectrums*len(fftfreqs), axis=1 ))


class RF_Modelised_Component(RF_Abstract_Modelised_Component):
    """Class representing a modelised RF component with frequency-dependent characteristics.

    Attributes:
        temp_kelvin (float): Temperature in Kelvin.
        freqs (Optional[np.ndarray]): Frequency points in Hz.
        gains_db (Optional[np.ndarray]): Gains in dB at each frequency.
        gains (Optional[np.ndarray]): Linear gains (complex if phases provided).
        nfs_db (Optional[np.ndarray]): Noise figures in dB at each frequency.
        nf__s (Optional[np.ndarray]): Linear noise figures.
        phases_rad (Optional[np.ndarray]): Phases in radians at each frequency.
        nominal_gain_for_im_db (float): Nominal gain for intermodulation calculations in dB.
        op1db_dbm (float): Output 1dB compression point in dBm.
        oip3_dbm (float): Output IP3 in dBm.
        iip2_dbm (float): Input IP2 in dBm.
        iip2 (float): Input IP2 voltage.
        oip3 (float): Output IP3 voltage.
        op1db (float): Output 1dB compression point voltage.
        a1 (float): Linear gain coefficient for distortion.
        a2 (float): Second-order non-linearity coefficient.
        k_oip3 (float): Scaling factor for third-order distortion.
        freqs_sup (np.ndarray): Supplementary frequencies for out-of-band (high).
        freqs_inf (np.ndarray): Inferior frequencies for out-of-band (low).
        gains_sup_db (np.ndarray): Supplementary gains in dB for out-of-band (high).
        gains_sup (np.ndarray): Supplementary linear gains for out-of-band (high).
        gains_inf (np.ndarray): Inferior linear gains for out-of-band (low).
        nfs___sup (np.ndarray): Supplementary noise figures for out-of-band (high).
        nfs___inf (np.ndarray): Inferior noise figures for out-of-band (low).
    """

    def __init__(self, freqs: np.ndarray, gains_db: np.ndarray, nfs_db: np.ndarray, phases_rad: np.ndarray | None = None,
                 op1ds_dbm: np.ndarray | None = None, iip3s_dbm: np.ndarray | None = None, iip2s_dbm: np.ndarray | None = None,
                 temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        """Initialize the modelised component with specified characteristics.

        Args:
            freqs: Frequency points in Hz.
            gains_db: Gains in dB at each frequency.
            nfs_db: Noise figures in dB at each frequency.
            phases_rad: Phases in radians.
            op1ds_dbm: Output 1dB compression point in dBm.
            iip3s_dbm: Input IP3 in dBm.
            iip2s_dbm: Input IP2 in dBm.
            temp_kelvin: Temperature in Kelvin.
        """
        # Initialize attributes belonging to input parameters
        self.temp_kelvin = temp_kelvin

        for _nm, _typs in (('freqs', (float, Iterable)), ('gains_db', (float, Iterable)),
                            ('nfs_db', (None, float, Iterable)), ('phases_rad', (None, float, Iterable)),
                            ('op1ds_dbm', (None, float, Iterable)), ('iip3s_dbm', (None, float, Iterable)), ('iip2s_dbm', (None, float, Iterable))):
            _convert = False
            #logger.info( f'<{self.__class__.__name__}> {_nm}, type: {type(eval(_nm))}, value: {eval(_nm)}' )

            for _typ in _typs:
                if _typ is float:
                    try:
                        setattr(self, _nm, np.array( [float(eval(_nm))] ))
                        _convert = True
                    except:
                        pass
                elif _typ is Iterable:
                    try:
                        setattr(self, _nm, np.array(eval(_nm)))
                        _convert = True
                    except:
                        pass
                elif _typ is None:
                    if eval(_nm) is None:
                        setattr(self, _nm, None)
                        _convert = True

                if _convert: break

            #logger.info( f'<{self.__class__.__name__}> {_nm}, type: {type(eval(_nm))}, value: {eval(_nm)}' )

            if not _convert:
                raise TypeError(f"Invalid type for {self.__class__.__name__}.{_nm}: expected {tuple(_typs)}, got {type(eval(_nm))}")
            elif getattr(self, _nm) is not None:
                if getattr(self, _nm).shape != self.freqs.shape:
                    raise ValueError(f"Invalid value for {self.__class__.__name__}.{_nm}: expected shape {self.freqs.shape}, got {getattr(self, _nm).shape}")
            elif _nm in ('nfs_db', 'phases_rad'):
                setattr(self, _nm, np.zeros_like(self.freqs))
            elif _nm in ('op1ds_dbm', 'iip3s_dbm'):
                setattr(self, _nm, infs_like(self.freqs))
            elif _nm == 'iip2s_dbm':
                # It seems likely that when IIP2 or OIP2 is not provided and IIP3 is known, there is no universal 
                # standard value to estimate, as IIP2 and IIP3 depend on specific component characteristics.
                # For RF mixers, research suggests that IIP2 is generally 20 to 40 dB higher than IIP3, with an average estimate around 25 dB.
                # For RF amplifiers, the difference between IIP2 and IIP3 is often 15 to 20 dB.
                # In practice, for modeling purposes, we can assume that IIP2 is roughly 25 dB higher than IIP3 for mixers, but this remains an approximation.
                setattr(self, _nm, self.iip3s_dbm + 25)  # Default IIP2 is 25 dB above IIP3

        # Initialize complementary attributes based on input parameters
        self.gains = gain_db_to_gain(self.gains_db) * np.exp(1j * self.phases_rad)  # Complex gains
        self.nf__s = nf_db_to_nf(self.nfs_db) 
        self.op1ds = dbm_to_voltage(self.op1ds_dbm)
        self.iip3s = dbm_to_voltage(self.iip3s_dbm)
        self.iip2s = dbm_to_voltage(self.iip2s_dbm)

        #for var_name in ('self.freqs', 'self.gains', 'self.nf__s', 'self.op1ds', 'self.iip3s', 'self.iip2s'):
            #logger.info( f'<{self.__class__.__name__}> {var_name}, {eval(var_name)}')

    def get_rf_parameters_adapted_to_signals(self, signals: Signals, temp_kelvin: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get frequency-dependent gains and noise figures for the signals.

        Args:
            signals: Input signals object.
            temp_kelvin: Temperature in Kelvin.

        Returns:
            Tuple of Frequencies, gains, noise figures, op1ds, iip3s, iip2s.
        """
        parameters = self.extend_rf_parameters(self.freqs, self.gains, self.nf__s, self.op1ds, self.iip3s, self.iip2s)

        return parameters
