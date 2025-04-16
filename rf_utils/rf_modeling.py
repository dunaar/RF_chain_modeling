#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project: RF_chain_modeling
RF Signal Simulation and Analysis Framework

This module provides a comprehensive framework for simulating and analyzing RF (Radio Frequency) signals and components.
It includes classes and functions for signal generation, processing, and visualization.

Key Features:
- Signal generation with tones and noise.
- RF component modeling (attenuators, amplifiers, cables, filters, antennas).
- Signal processing and analysis.
- Visualization of temporal and spectral representations.

Author: Pessel Arnaud
Date: 2025-03-15
Version: 0.1
GitHub: https://github.com/dunaar/RF_chain_modeling
License: MIT
"""

__version__ = "0.1"

from cProfile import label
import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable
from matplotlib.pylab import f
from tqdm import tqdm
from typing import Optional, Tuple, Union

import numpy as np
from numpy import pi

from scipy.signal import butter, freqz
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from .util_search_mediane import search_mediane_for_slope

# ====================================================================================================
# Constants
# ====================================================================================================

# Physical Constants
K_B = 1.38e-23  # Boltzmann constant in Joules per Kelvin

# Default Parameters
DEFAULT_TEMP_KELVIN = 298.15  # Default temperature in Kelvin: 298,15K = 25°C
DEFAULT_IMPED_OHMS = 50.0  # Default impedance in ohms for RF systems
DEFAULT_N_WINDOWS = 32  # Default number of signal windows for processing

# ====================================================================================================
# Utility Functions
# ====================================================================================================
def infs_like(arr: np.ndarray) -> np.ndarray:
    return np.full_like(arr, np.inf)  # Create an array of the same shape as arr filled with infinity

def dbm_to_watts(power_dbm: float) -> float:
    """Convert power from dBm to watts.
    
    Args:
        power_dbm (float): Power in dBm (decibels relative to 1 milliwatt).
    
    Returns:
        float: Power in watts.
    """
    return 10 ** (power_dbm / 10) / 1000  # Convert dBm to milliwatts, then to watts

def watts_to_voltage(power_watts: float, imped_ohms: float = DEFAULT_IMPED_OHMS) -> float:
    """Convert power from watts to voltage assuming a given impedance.
    
    Args:
        power_watts (float): Power in watts.
        imped_ohms (float): Impedance in ohms, defaults to 50 ohms.
    
    Returns:
        float: Voltage.
    """
    return np.sqrt(power_watts * imped_ohms)  # V = sqrt(P * R) based on Ohm's law

def dbm_to_voltage(power_dbm: float, imped_ohms: float = DEFAULT_IMPED_OHMS) -> float:
    """Convert power from dBm to voltage.
    
    Args:
        power_dbm (float): Power in dBm.
        imped_ohms (float): Impedance in ohms, defaults to 50 ohms.
    
    Returns:
        float: Voltage.
    """
    return watts_to_voltage(dbm_to_watts(power_dbm), imped_ohms)  # Chain conversion: dBm -> watts -> voltage

def gain_db_to_gain(gain_db: float) -> float:
    """Convert gain from dB to linear scale (voltage gain).
    
    Args:
        gain_db (float): Gain in decibels.
    
    Returns:
        float: Linear voltage gain.
    """
    return 10 ** (gain_db / 20.0)  # Voltage gain: G = 10^(dB/20)

def gain_to_gain_db(gain: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert gain from linear scale to dB.
    
    Args:
        gain (Union[float, np.ndarray]): Linear gain value (scalar or array).
    
    Returns:
        Union[float, np.ndarray]: Gain in dB.
    """
    return 20 * np.log10(np.abs(gain))  # dB = 20 * log10(|G|) for voltage gain

def nf_db_to_nf(nf_db: float) -> float:
    """Convert noise figure from dB to linear scale.
    
    Args:
        nf_db (float): Noise figure in dB.
    
    Returns:
        float: Linear noise figure contribution (standard deviation of noise voltage).
    """
    return np.sqrt(10 ** (nf_db / 10) - 1)  # NF_linear = sqrt(F - 1), where F = 10^(NF_dB/10)

def nf_to_nf_db(nf: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert noise figure from linear scale to dB.
    
    Args:
        nf (Union[float, np.ndarray]): Linear noise figure (scalar or array).
    
    Returns:
        Union[float, np.ndarray]: Noise figure in dB.
    """
    return 10 * np.log10(nf ** 2 + 1)  # NF_dB = 10 * log10(F), where F = NF_linear^2 + 1

def mul_nfs(nf1: Union[float, np.ndarray], nf2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Multiply noise factors to compute combined noise figure using Friis formula.
    
    Args:
        nf1 (Union[float, np.ndarray]): Linear noise figure of first component.
        nf2 (Union[float, np.ndarray]): Linear noise figure of second component.
    
    Returns:
        Union[float, np.ndarray]: Combined linear noise figure.
    """
    return np.sqrt((nf1 ** 2 + 1) * (nf2 ** 2 + 1) - 1)  # Friis formula for noise factor multiplication

def voltage_to_watts(voltage: Union[float, np.ndarray], imped_ohms: float = 50) -> Union[float, np.ndarray]:
    """Convert voltage to power in watts.
    
    Args:
        voltage (Union[float, np.ndarray]): Voltage value (scalar or array).
        imped_ohms (float): Impedance in ohms, defaults to 50 ohms.
    
    Returns:
        Union[float, np.ndarray]: Power in watts.
    """
    return voltage ** 2 / imped_ohms  # P = V^2 / R

def watts_to_dbm(power_watts: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert power from watts to dBm.
    
    Args:
        power_watts (Union[float, np.ndarray]): Power in watts (scalar or array).
    
    Returns:
        Union[float, np.ndarray]: Power in dBm.
    """
    return 10 * np.log10(power_watts * 1000)  # dBm = 10 * log10(P * 1000) to convert watts to milliwatts

def voltage_to_dbm(voltage: Union[float, np.ndarray], imped_ohms: float = DEFAULT_IMPED_OHMS) -> Union[float, np.ndarray]:
    """Convert voltage to power in dBm.
    
    Args:
        voltage (Union[float, np.ndarray]): RMS voltage.
        imped_ohms (float): Impedance in ohms, defaults to 50 ohms.
    
    Returns:
        Union[float, np.ndarray]: Power in dBm.
    """
    return watts_to_dbm(voltage_to_watts(voltage, imped_ohms))  # Chain conversion: voltage -> watts -> dBm

def calculate_rms(signal: np.ndarray) -> float:
    """Calculate the Root Mean Square (RMS) value of a signal.
    
    Args:
        signal (np.ndarray): Input signal (1D or 2D array).
    
    Returns:
        float: RMS value of the signal (scalar for 1D, array for 2D).
    """
    return np.sqrt(np.mean(np.abs(signal) ** 2))  # RMS = sqrt(mean(|signal|^2))

def calculate_rms_dbm(signal: np.ndarray) -> float:
    """Calculate the RMS value of a signal in dBm.
    
    Args:
        signal (np.ndarray): Input signal (1D or 2D array).
    
    Returns:
        float: RMS power in dBm.
    """
    return voltage_to_dbm(calculate_rms(signal))  # Convert RMS voltage to dBm

def thermal_noise_power_dbm(temp_kelvin: float, bw_hz: float) -> float:
    """Calculate thermal noise power in dBm based on temperature and bandwidth.
    
    Args:
        temp_kelvin (float): Temperature in Kelvin.
        bw_hz (float): Bandwidth in Hertz.
    
    Returns:
        float: Thermal noise power in dBm.
    """
    return watts_to_dbm(K_B * temp_kelvin * bw_hz)  # P = K_B * T * B, then convert to dBm

# ====================================================================================================
# Signal Processing Functions
# ====================================================================================================

def compute_spectrums(sigxd: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the frequency spectrum of a signal using FFT.
    
    Args:
        sigxd (np.ndarray): Input signal (1D or 2D array).
        sampling_rate (float): Sampling rate in Hz.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Frequency bins and corresponding spectrums.
    """
    if len(sigxd.shape) == 1:
        sig2d = sigxd.reshape(1, sigxd.shape[0])  # Convert 1D to 2D for consistent processing
    else:
        sig2d = sigxd

    n_points = sig2d.shape[1]  # Number of samples in each window

    freqs = np.fft.fftfreq(n_points, 1 / sampling_rate)  # Frequency bins
    spectrums = np.fft.fft(sig2d, axis=1) / n_points  # FFT normalized by number of points

    return freqs, spectrums

def get_spectrums_power_n_phase(freqs: np.ndarray, spectrums: np.ndarray, imped_ohms: float = DEFAULT_IMPED_OHMS) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the power and phase spectrum from FFT results.
    
    Args:
        freqs (np.ndarray): Frequency bins.
        spectrums (np.ndarray): Complex FFT spectrums (1D or 2D).
        imped_ohms (float): Impedance in ohms, defaults to 50 ohms.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Power spectrum in dBm and phase spectrum in radians.
    """
    spects_amp = np.abs(spectrums)  # Amplitude of the spectrum
    spects_power = voltage_to_dbm(spects_amp, imped_ohms)  # Convert amplitude to power in dBm
    spects_phase = np.angle(spectrums)  # Phase in radians
    return spects_power, spects_phase

# ====================================================================================================
# Visualization Functions
# ====================================================================================================

def plot_temporal_signal(time: np.ndarray, sigxd: np.ndarray, tmin: Optional[float] = None, tmax: Optional[float] = None, title: str = "Temporal Signal", ylabel: str = "Amplitude") -> None:
    """Plot the temporal representation of a signal with automatic unit scaling.
    
    Args:
        time (np.ndarray): Time array in seconds.
        sigxd (np.ndarray): Signal array (1D or 2D).
        tmin (Optional[float]): Minimum time to plot, defaults to signal start.
        tmax (Optional[float]): Maximum time to plot, defaults to signal end.
        title (str): Plot title, defaults to "Temporal Signal".
        ylabel (str): Y-axis label, defaults to "Amplitude".
    """
    tmin = tmin if tmin is not None else 0  # Default to start of signal
    tmax = tmax if tmax is not None else time.max()  # Default to end of signal
    idx_min = np.argmin(np.abs(time - tmin))  # Index of closest time to tmin
    idx_max = np.argmin(np.abs(time - tmax))  # Index of closest time to tmax

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

def plot_signal_spectrum(freqs: np.ndarray, spectrum_power: np.ndarray, spectrum_phase: Optional[np.ndarray] = None, title_power: str = "Power Spectrum", title_phase: str = "Phase Spectrum", ylabel_power: str = "Power (dBm)", ylabel_phase: str = "Phase (radians)") -> None:
    """Plot the frequency power spectrum and optionally the phase spectrum with automatic unit scaling.
    
    Args:
        freqs (np.ndarray): Frequency bins in Hz.
        spectrum_power (np.ndarray): Power spectrum in dBm (1D or 2D).
        spectrum_phase (Optional[np.ndarray]): Phase spectrum in radians (1D or 2D), defaults to None.
        title_power (str): Title for power spectrum plot, defaults to "Power Spectrum".
        title_phase (str): Title for phase spectrum plot, defaults to "Phase Spectrum".
        ylabel_power (str): Y-axis label for power spectrum, defaults to "Power (dBm)".
        ylabel_phase (str): Y-axis label for phase spectrum, defaults to "Phase (radians)".
    """
    freqs = np.array(freqs)
    idx_max = np.argmax(freqs) + 1  # Index of maximum positive frequency

    # Determine appropriate frequency unit for plotting
    if freqs.max() - freqs[0] > 10e9:
        freqs = freqs / 1e9
        unit = 'GHz'
    elif freqs.max() - freqs[0] > 10e6:
        freqs = freqs / 1e6
        unit = 'MHz'
    elif freqs.max() - freqs[0] > 10e3:
        freqs = np.array(freqs) / 1e3
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
        bw_hz (float): Bandwidth in Hz (half of sampling rate).
        shape (Tuple[int, int]): Shape of the signal array (n_windows, n_points).
        time (np.ndarray): Time array for each window.
        sig2d (np.ndarray): 2D array of signals (windows x samples).
        _spectrum_uptodate (bool): Flag indicating if spectrum is up-to-date.
        _spectrums (Optional[np.ndarray]): Cached FFT spectrums.
        _spects_power (Optional[np.ndarray]): Cached power spectrums in dBm.
        _spects_phase (Optional[np.ndarray]): Cached phase spectrums in radians.
    """

    def __init__(self, fmax: float, bin_width: float, n_windows: int = DEFAULT_N_WINDOWS, imped_ohms: float = DEFAULT_IMPED_OHMS, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        """Initialize the Signals object with specified parameters.
        
        Args:
            fmax (float): Maximum frequency of signals in Hz.
            bin_width (float): Width of the spectral bins in Hz.
            n_windows (int): Number of signal windows, defaults to 32.
            imped_ohms (float): Impedance in ohms, defaults to 50 ohms.
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K (25°C).
        """
        self.fmax = fmax
        self.bin_width = bin_width
        self.imped_ohms = imped_ohms
        self.temp_kelvin = temp_kelvin
        self.n_windows = n_windows

        self.duration = 1 / bin_width  # Duration is inverse of bin width
        self.n_points = int(np.ceil(2.2 * fmax / bin_width))  # Ensure sufficient sampling
        self.sampling_rate = self.n_points / self.duration  # Sampling rate in Hz
        self.freqs = np.fft.fftfreq(self.n_points, 1 / self.sampling_rate)  # Frequency bins

        self.bw_hz = self.sampling_rate / 2  # Nyquist bandwidth

        self.shape = (self.n_windows, self.n_points)  # Shape of signal array

        self.time = np.linspace(0, self.duration, self.n_points, endpoint=False)  # Time array
        self.sig2d = Signals.generate_noise_dbm(self.shape, -1000)  # Initialize with very low noise

        self._spectrum_uptodate = False
        self._spectrums         = None
        self._spects_power      = None
        self._spects_phase      = None

    @staticmethod
    def generate_signal_dbm(time: np.ndarray, freq: float, power_dbm: float, phase: float, imped_ohms: float = DEFAULT_IMPED_OHMS) -> np.ndarray:
        """Generate a monotone sinusoidal signal with specified power and phase.
        
        Args:
            time (np.ndarray): Time array in seconds.
            freq (float): Frequency in Hz.
            power_dbm (float): Power in dBm.
            phase (float): Phase in radians.
            imped_ohms (float): Impedance in ohms, defaults to 50 ohms.
        
        Returns:
            np.ndarray: Generated signal array.
        """
        amp_rms = dbm_to_voltage(power_dbm, imped_ohms)  # RMS amplitude
        amp_pk = np.sqrt(2) * amp_rms  # Peak amplitude from RMS (for sine wave)
        return amp_pk * np.sin(2 * pi * freq * time + phase)  # Generate sine wave

    @staticmethod
    def generate_noise_dbm(shape: Union[int, Tuple[int, ...]], power_dbm: float, imped_ohms: float = DEFAULT_IMPED_OHMS) -> np.ndarray:
        """Generate Gaussian noise with specified power in dBm.
        
        Args:
            shape (Union[int, Tuple[int, ...]]): Shape of the noise array (scalar or tuple).
            power_dbm (float): Power in dBm.
            imped_ohms (float): Impedance in ohms, defaults to 50 ohms.
        
        Returns:
            np.ndarray: Generated noise array.
        """
        amp = dbm_to_voltage(power_dbm, imped_ohms)  # amplitude
        return np.random.normal(0, amp, shape)  # Gaussian noise with zero mean

    def compute_spectrum(self, force: bool = False) -> None:
        """Compute the frequency spectrum of the signals if not up-to-date.
        
        Args:
            force (bool): Force recomputation even if spectrum is up-to-date, defaults to False.
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
            freq (float): Target frequency in Hz.
        
        Returns:
            int: Index of the closest frequency bin.
        """
        return np.argmin(np.abs(self.freqs - freq))

    def add_signal(self, sigxd: np.ndarray) -> None:
        """Add a signal to the existing signals.
        
        Args:
            sigxd (np.ndarray): Signal to add (1D or 2D array).
        
        Raises:
            ValueError: If the shape of sigxd is incompatible with self.sig2d.
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
            freq (float): Frequency in Hz.
            power_dbm (float): Power in dBm.
            phase (float): Phase in radians.
        """
        tone = self.generate_signal_dbm(self.time, freq, power_dbm, phase, self.imped_ohms)
        self.sig2d += tone
        self._spectrum_uptodate = False

    def add_noise(self, power_dbm: float) -> None:
        """Add Gaussian noise to the signals.
        
        Args:
            power_dbm (float): Power in dBm.
        """
        noise = self.generate_noise_dbm(self.shape, power_dbm, self.imped_ohms)
        self.sig2d += noise
        self._spectrum_uptodate = False

    def add_thermal_noise(self, temp_kelvin: Optional[float] = None) -> None:
        """Add thermal noise to the signals based on temperature and bandwidth.
        
        Args:
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to instance temperature.
        """
        temp_kelvin = temp_kelvin if temp_kelvin is not None else self.temp_kelvin
        noise_power_dbm = thermal_noise_power_dbm(temp_kelvin, self.bw_hz)
        self.add_noise(noise_power_dbm)

    def rms_dbm(self) -> float:
        """Calculate the RMS value of the signals in dBm.
        
        Returns:
            float: RMS power in dBm.
        """
        return voltage_to_dbm(np.sqrt(np.mean(np.abs(self.sig2d) ** 2)), self.imped_ohms)

    def plot_temporal(self, tmin: Optional[float] = None, tmax: Optional[float] = None, title: str = "Temporal Signal", ylabel: str = "Amplitude") -> None:
        """Plot the temporal representation of the signals.
        
        Args:
            tmin (Optional[float]): Minimum time to plot, defaults to signal start.
            tmax (Optional[float]): Maximum time to plot, defaults to signal end.
            title (str): Plot title, defaults to "Temporal Signal".
            ylabel (str): Y-axis label, defaults to "Amplitude".
        """
        plot_temporal_signal(self.time, self.sig2d, tmin, tmax, title, ylabel)

    def plot_spectrum(self, title_power: str = "Power Spectrum", title_phase: str = "Phase Spectrum", ylabel_power: str = "Power (dBm)", ylabel_phase: str = "Phase (radians)") -> None:
        """Plot the frequency spectrum of the signals (power and phase).
        
        Args:
            title_power (str): Title for power spectrum plot, defaults to "Power Spectrum".
            title_phase (str): Title for phase spectrum plot, defaults to "Phase Spectrum".
            ylabel_power (str): Y-axis label for power spectrum, defaults to "Power (dBm)".
            ylabel_phase (str): Y-axis label for phase spectrum, defaults to "Phase (radians)".
        """
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
            x (np.ndarray): Input signal array.
            k (float): Scaling factor for amplitude limiting.
        
        Returns:
            np.ndarray: Transformed signal array.
        """
        return np.tanh(x / k) * k

    @abstractmethod
    def process_signals(self, signals: Signals, temp_kelvin: Optional[float] = None) -> None:
        """Abstract method to process signals. Must be implemented by subclasses.
        
        Args:
            signals (Signals): Input signals object to process.
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to None.
        """
        pass

    def process(self, signals: Signals, temp_kelvin: Optional[float] = None, inplace: bool = True) -> Optional[Signals]:
        """Process signals through the RF component, optionally returning a new object.
        
        Args:
            signals (Signals): Input signals object.
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to None.
            inplace (bool): If True, modify signals in place; if False, return a new Signals object.
        
        Returns:
            Optional[Signals]: Processed signals if inplace is False, otherwise None.
        """
        if not inplace:
            signals = copy.deepcopy(signals)

        means = (1.0 - 1e-3) * signals.sig2d.mean(axis=1)  # Calculate DC offset
        signals.sig2d = signals.sig2d - means[:, np.newaxis]  # DC block: remove continuous signal

        self.process_signals(signals, temp_kelvin=temp_kelvin)
        signals._spectrum_uptodate = False

        return signals if not inplace else None

    def assess_gain(self, fmin: float = 400e6, fmax: float = 19e9, step: float = 100e6, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Assess the gain, phase, and noise figure versus frequency of the RF component.
        
        Args:
            fmin (float): Minimum frequency in Hz, defaults to 400 MHz.
            fmax (float): Maximum frequency in Hz, defaults to 19 GHz.
            step (float): Frequency step in Hz, defaults to 100 MHz.
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Frequencies, gains (dB), phases (radians), noise figures (dB).
        """
        print("""Assess the gain and phase versus frequency of the RF component.""")
        
        bin_width = step / 2
        n_windows = 128

        # Initialize noisy signals
        noisy_signals = Signals(fmax, bin_width, n_windows=n_windows, imped_ohms=50, temp_kelvin=temp_kelvin)
        noisy_signals.add_thermal_noise(temp_kelvin=temp_kelvin)

        # Frequencies for testing
        freqs_for_test = np.linspace(fmin, fmax, int((fmax - fmin) / step) + 1)
        gains = np.zeros_like(freqs_for_test)
        phass = np.zeros_like(freqs_for_test)
        n_fgs = np.zeros_like(freqs_for_test)

        for idx_frq, freq in tqdm(enumerate(freqs_for_test), total=len(freqs_for_test), desc="Assessing Gain"):
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

        #for var in ('freqs_for_test', 'gains', 'phass, n_fgs'):
        #    print('%s: '%(var), eval(var))

        return freqs_for_test, gains, phass, n_fgs

    def assess_ipx_for_freq(self, fc: float = 9e9, df: float = 100e6, temp_kelvin: float = DEFAULT_TEMP_KELVIN, toplot: bool = False) -> Tuple[float, float, float, float]:
        """Assess the IP2 and IP3 (Intercept Points of order 2 and 3) for a specific frequency.
        
        Args:
            fc (float): Center frequency in Hz, defaults to 9 GHz.
            df (float): Frequency separation in Hz, defaults to 400 MHz.
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
            toplot (bool): If True, generate plots for the assessment.
        
        Returns:
            Tuple[float, float, float, float]: Gain (dB), OP1dB (dBm), IIP2 (dBm), OIP3 (dBm).
        """
        print("## Assess the IP2 and IP3 (Intercept Point of order 2 and 3) of the RF component.")
        print("   -- Central frequency: %.3f GHz, df: %.3f MHz" % (fc / 1e9, df / 1e6), end='')
        
        # --------------------------------------------------------
        fmax = 3. * fc
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
        print(", F1: %.3f GHz, F2: %.3f GHz" % (f1 / 1e9, f2 / 1e9))
        
        # Indexes of frequencies in spectrum
        signals = Signals(fmax, bin_width, n_windows=n_windows, imped_ohms=50, temp_kelvin=temp_kelvin)

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
            print("Error: Unable to characterize Gain finely.")
            gain_db  = outpt_pwr_m[len(outpt_pwr_m)//2] - input_pwr_m[len(outpt_pwr_m)//2]
    
        if op1db_dbm is None:
            print("Error: Unable to characterize P1dB finely.")
            ip1db_dbm = input_pwr_m[-1]
            op1db_dbm = outpt_pwr_m[-1]
        
        input_pwr_m = np.array(input_pwr_m)
        outpt_pwr_m = np.array(outpt_pwr_m)
        h1_slope, h1_inter = search_mediane_for_slope(input_pwr_m, outpt_pwr_m, 1)
        print('', h1_slope, h1_inter)
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
        if np.count_nonzero((np.abs(im3___power - (im3_slope * input_pwr_d + im3_inter)) < 1.))/len(input_pwr_d) > 5e-2:
            iip3_dbm = (gain_db - im3_inter) / 2
        else:
            print("Error: Unable to characterize IP3 finely.")
            iip3_dbm = op1db_dbm + 14 - gain_db
        oip3_dbm  = iip3_dbm + gain_db

        # Retrieving IP2
        im2_slope, im2_inter = search_mediane_for_slope(input_pwr_d, im2___power, 2)
        if np.count_nonzero((np.abs(im2___power - (im2_slope * input_pwr_d + im2_inter)) < 1.))/len(input_pwr_d) > 5e-2:
            iip2_dbm = gain_db - im2_inter
        else:
            print("Error: Unable to characterize IP2 finely.")
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
                print('%s: '%(var_name), eval(var_name))
        # --------------------------------------------------------
        
        return gain_db, op1db_dbm, iip3_dbm, oip3_dbm, iip2_dbm, oip2_dbm
    
    def assess_ipx(self, freq_min: float, freq_max: float, fr_stp: float = 1e9, df: float = 200e6, temp_kelvin: float = DEFAULT_TEMP_KELVIN, toplot: bool = False) -> list:
        """Assess the IP2 and IP3 for a range of frequencies.
        
        Args:
            freq_min (float): Minimum frequency in Hz.
            freq_max (float): Maximum frequency in Hz.
            fr_stp (float): Frequency step in Hz, defaults to 1 GHz.
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
        
        Returns:
            list: List of tuples containing (frequency, (gain_db, op1db_dbm, iip2_dbm, oip3_dbm)).
        """
        results = []
        freqs = np.arange(freq_min, freq_max+0.1*fr_stp, fr_stp) # freq_max+0.1*fr_stp: in order to include freq_max
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

    def __init__(self, rf_components: list = []) -> None:
        """Initialize the RF chain with a list of components.
        
        Args:
            rf_components (list): List of RF_Abstract_Base_Component objects, defaults to empty list.
        """
        self.rf_components = list(rf_components)

    def process_signals(self, signals: Signals, temp_kelvin: Optional[float] = None) -> None:
        """Process signals through each RF component in the chain sequentially.
        
        Args:
            signals (Signals): Input signals object.
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to None.
        """
        for rf_compnt in self.rf_components:
            rf_compnt.process(signals, temp_kelvin)

# ====================================================================================================
# Attenuator Class
# ====================================================================================================

class Attenuator(RF_Abstract_Base_Component):
    """Class representing an attenuator RF component.
    
    Attributes:
        temp_kelvin (float): Temperature in Kelvin.
        att_db (float): Attenuation in dB (positive value).
        nf_db (float): Noise figure in dB (equals attenuation for passive device).
        gain (float): Linear gain (less than 1 due to attenuation).
        nf (float): Linear noise figure contribution.
    """

    def __init__(self, att_db: float, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        """Initialize the attenuator with specified attenuation.
        
        Args:
            att_db (float): Attenuation in dB (positive value).
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
        """
        self.temp_kelvin = temp_kelvin
        self.att_db = att_db
        self.nf_db = att_db  # Noise figure equals attenuation for a passive attenuator

        self.gain = gain_db_to_gain(-att_db)  # Convert attenuation to linear gain (< 1)
        self.nf = nf_db_to_nf(self.nf_db)

    def process_signals(self, signals: Signals, temp_kelvin: Optional[float] = None) -> None:
        """Process signals by applying attenuation and adding thermal noise.
        
        Args:
            signals (Signals): Input signals object.
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to instance temperature.
        """
        temp_kelvin = temp_kelvin if temp_kelvin is not None else self.temp_kelvin

        # Generate thermal noise based on temperature and bandwidth
        noise = Signals.generate_noise_dbm(signals.shape, thermal_noise_power_dbm(temp_kelvin, signals.bw_hz), signals.imped_ohms)
        signals.sig2d += self.nf * noise  # Add noise contribution
        signals.sig2d *= self.gain  # Apply attenuation

# ====================================================================================================
# Simple Amplifier Class
# ====================================================================================================

class Simple_Amplifier(RF_Abstract_Base_Component):
    """Class representing a simple amplifier with gain, noise figure, and non-linearities.
    
    Attributes:
        temp_kelvin (float): Temperature in Kelvin.
        gain_db (float): Gain in dB.
        nf_db (float): Noise figure in dB.
        op1db_dbm (float): Output 1dB compression point in dBm.
        oip3_dbm (float): Output third-order intercept point (IP3) in dBm.
        iip2_dbm (float): Input second-order intercept point (IP2) in dBm.
        gain (float): Linear gain.
        nf (float): Linear noise figure contribution.
        iip2 (float): Input IP2 voltage.
        oip3 (float): Output IP3 voltage.
        op1db (float): Output 1dB compression point voltage.
        a1 (float): Linear gain coefficient.
        a2 (float): Second-order non-linearity coefficient.
        k_oip3 (float): Scaling factor for third-order distortion limiting.
    """

    def __init__(self, gain_db: float, nf_db: float, op1db_dbm: float = 20, oip3_dbm: float = 10, iip2_dbm: float = 40, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        """Initialize the amplifier with specified parameters.
        
        Args:
            gain_db (float): Gain in dB.
            nf_db (float): Noise figure in dB.
            op1db_dbm (float): Output 1dB compression point in dBm, defaults to 20 dBm.
            oip3_dbm (float): Output IP3 in dBm, defaults to 10 dBm.
            iip2_dbm (float): Input IP2 in dBm, defaults to 40 dBm.
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
        """
        self.temp_kelvin = temp_kelvin
        self.gain_db     = gain_db
        self.nf_db       = nf_db
        self.op1db_dbm   = op1db_dbm
        self.oip3_dbm    = oip3_dbm
        self.iip2_dbm    = iip2_dbm

        # Convert to linear scale
        self.gain = gain_db_to_gain(self.gain_db)
        self.nf   = nf_db_to_nf(self.nf_db)

        self.iip2  = dbm_to_voltage(self.iip2_dbm)
        self.oip3  = dbm_to_voltage(self.oip3_dbm)
        self.op1db = dbm_to_voltage(self.op1db_dbm)

        self.a1     = self.gain
        self.a2     = -0.43 * self.a1 / self.iip2  # Second-order coefficient based on IIP2
        self.k_oip3 = 6.5e-1 * self.oip3           # Scaling factor for third-order distortion
    
    def process_signals(self, signals: Signals, temp_kelvin: Optional[float] = None) -> None:
        """Process signals by applying gain, noise, and non-linear distortions.
        
        Args:
            signals (Signals): Input signals object.
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to instance temperature.
        """
        temp_kelvin = temp_kelvin if temp_kelvin is not None else self.temp_kelvin

        # Add thermal noise
        noise = Signals.generate_noise_dbm(signals.shape, thermal_noise_power_dbm(temp_kelvin, signals.bw_hz), signals.imped_ohms)
        signals.sig2d += self.nf * noise

        s   = signals.sig2d

        # Apply linear gain with third-order distortion limiting
        s1  = self.ft(self.a1 * s, self.k_oip3)

        # Apply second-order non-linearity and remove DC component
        s_2  = self.a2 * s ** 2
        s_2 -= s_2.mean(1)[:, np.newaxis]
        s_2  = self.ft(s_2, self.op1db * 1.2)
        
        # Combine effects with final compression limiting
        signals.sig2d = self.ft(s1+s_2, self.op1db*6)
        
        #signals.sig2d = self.ft(self.a1 * signals.sig2d, self.k_oip3) + self.a2 * self.ft(signals.sig2d, self.op1db * 2) ** 2
        # OP1dB: 1dB compression point output power
        #signals.sig2d = self.ft(signals.sig2d, self.op1db * 6)

# ====================================================================================================
# RF Modelised Component Class
# ====================================================================================================
switch_print = True
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
    def get_rf_parameters_adapted_to_signals(self, signals: Signals, temp_kelvin: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
                             op1ds: np.ndarray, iip3s: np.ndarray, iip2s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    
    def process_signals(self, signals: Signals, temp_kelvin: Optional[float] = None) -> None:
        """Process signals by applying frequency-dependent gains, noise, and distortion.
        
        Args:
            signals (Signals): Input signals object.
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to instance temperature.
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
            print(np.count_nonzero(freqs<0.), np.count_nonzero(freqs>0.))
            print(freqs[freqs<0.], freqs[freqs>0.])
            raise ValueError("Frequency-dependent parameters are not set correctly.")

        #for var_name in ('freqs', 'gains', 'nf__s', 'op1ds', 'iip3s', 'iip2s'):
        #    print(var_name, eval(var_name))

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
        noise = Signals.generate_noise_dbm(signals.shape, thermal_noise_power_dbm(temp_kelvin, signals.bw_hz), signals.imped_ohms)
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

        #print('out:', voltage_to_dbm(np.abs(spectrums[:, idx_frtest])))

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

    def __init__(self, freqs: np.ndarray, gains_db: np.ndarray, nfs_db: np.ndarray, phases_rad: Optional[np.ndarray] = None,
                 op1ds_dbm: Optional[np.ndarray] = None, iip3s_dbm: Optional[np.ndarray] = None, iip2s_dbm: Optional[np.ndarray] = None,
                 temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        """Initialize the modelised component with specified characteristics.
        
        Args:
            freqs (np.ndarray): Frequency points in Hz.
            gains_db (np.ndarray): Gains in dB at each frequency.
            nfs_db (np.ndarray): Noise figures in dB at each frequency.
            phases_rad (Optional[np.ndarray]): Phases in radians, defaults to None.
            nominal_gain_for_im_db (Optional[float]): Nominal gain for IM in dB, defaults to None.
            op1db_dbm (float): Output 1dB compression point in dBm, defaults to infinity.
            oip3_dbm (float): Output IP3 in dBm, defaults to infinity.
            iip2_dbm (float): Input IP2 in dBm, defaults to infinity.
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
        """
        # Initialize attributes belonging to input parameters
        self.temp_kelvin = temp_kelvin

        for _nm, _typs in (('freqs', (float, Iterable)), ('gains_db', (float, Iterable)),
                            ('nfs_db', (None, float, Iterable)), ('phases_rad', (None, float, Iterable)),
                            ('op1ds_dbm', (None, float, Iterable)), ('iip3s_dbm', (None, float, Iterable)), ('iip2s_dbm', (None, float, Iterable))):
            _convert = False
            #print(_nm, type(eval(_nm)), eval(_nm))

            for _typ in _typs:
                if _typ is float:
                    try:
                        setattr(self, _nm, np.array( [float(eval(_nm))] ))
                        _convert = True
                    except: pass
                elif _typ is Iterable:
                    try:
                        setattr(self, _nm, np.array(eval(_nm)))
                        _convert = True
                    except: pass
                elif _typ is None:
                    if eval(_nm) is None:
                        setattr(self, _nm, None)
                        _convert = True

                if _convert: break

            #print('self.'+_nm, type(eval('self.'+_nm)), eval('self.'+_nm))

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
                # Il semble probable que, lorsqu'IIP2 ou OIP2 n'est pas fourni et que l'IIP3 est connu, il n'existe pas de valeur standard universelle à estimer, car IIP2 et IIP3 dépendent des caractéristiques spécifiques du composant.
                # Pour les mélangeurs RF, des recherches suggèrent que l'IIP2 est généralement 20 à 40 dB plus élevé que l'IIP3, avec une estimation moyenne autour de 25 dB.
                # Pour les amplificateurs RF, la différence entre IIP2 et IIP3 est souvent de 15 à 20 dB.
                # En pratique, pour la modélisation, on peut supposer que l'IIP2 est environ 25 dB plus élevé que l'IIP3 pour les mélangeurs, mais cela reste une approximation.
                setattr(self, _nm, self.iip3s_dbm + 25)  # Default IIP2 is 25 dB above IIP3
        
        # Initialize complementary attributes based on input parameters
        self.gains = gain_db_to_gain(self.gains_db) * np.exp(1j * self.phases_rad)  # Complex gains
        self.nf__s = nf_db_to_nf(self.nfs_db) 
        self.op1ds = dbm_to_voltage(self.op1ds_dbm)
        self.iip3s = dbm_to_voltage(self.iip3s_dbm)
        self.iip2s = dbm_to_voltage(self.iip2s_dbm)

        #for var_name in ('self.freqs', 'self.gains', 'self.nf__s', 'self.op1ds', 'self.iip3s', 'self.iip2s'):
        #    print(var_name, eval(var_name))

    def get_rf_parameters_adapted_to_signals(self, signals: Signals, temp_kelvin: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get frequency-dependent gains and noise figures for the signals.
        
        Args:
            signals (Signals): Input signals object.
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to instance temperature.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Frequencies, gains, noise figures.
        
        Raises:
            ValueError: If frequency-dependent parameters are not set.
        """
        parameters = self.extend_rf_parameters(self.freqs, self.gains, self.nf__s, self.op1ds, self.iip3s, self.iip2s)

        return parameters

# ====================================================================================================
# RF Cable Class
# ====================================================================================================

class RF_Cable(RF_Abstract_Modelised_Component):
    """Class representing an RF cable with frequency-dependent insertion losses.
    
    Attributes:
        length_m (float): Cable length in meters.
        alpha (float): Attenuation coefficient (dB/sqrt(Hz)).
        insertion_losses_dB (float): Fixed insertion loss in dB.
    """

    def __init__(self, length_m: float, alpha: float = 5.2e-06, insertion_losses_dB: float = 0, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        """Initialize the RF cable with specified length and losses.
        
        Args:
            length_m (float): Cable length in meters.
            alpha (float): Attenuation coefficient in dB/sqrt(Hz), defaults to 5.2e-06.
            insertion_losses_dB (float): Fixed insertion loss in dB, defaults to 0.
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
        """
        self.temp_kelvin = temp_kelvin
        self.length_m = length_m
        self.alpha = alpha
        self.insertion_losses_dB = insertion_losses_dB

    def get_rf_parameters_adapted_to_signals(self, signals: Signals, temp_kelvin: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get frequency-dependent gains and noise figures for the RF cable.
        
        Args:
            signals (Signals): Input signals object.
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to instance temperature.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Frequencies, gains, noise figures.
        """
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin

        freqs    = signals.freqs[signals.freqs >= 0]
        gains_db = -self.insertion_losses_dB - self.alpha * np.sqrt(freqs) * self.length_m  # Frequency-dependent loss
        gains    = gain_db_to_gain(gains_db)
        nf__s    = nf_db_to_nf(-gains_db)  # Noise figure equals negative gain for passive device

        return freqs, gains, nf__s, infs_like(freqs), infs_like(freqs), infs_like(freqs)  # No op1db, iip3, iip2 for passive device

# ====================================================================================================
# High Pass Filter Class
# ====================================================================================================

class HighPassFilter(RF_Abstract_Modelised_Component):
    """Class representing a high-pass filter with specified cutoff frequency and order.
    
    Attributes:
        cutoff_freq (float): Cutoff frequency in Hz.
        order (int): Filter order.
        q_factor (float): Quality factor (not used in Butterworth design).
        insertion_losses_dB (float): Insertion losses in dB.
        gain_losses (float): Linear gain factor due to insertion losses.
    """

    def __init__(self, cutoff_freq: float, order: int = 1, q_factor: float = 1, insertion_losses_dB: float = 0, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        """Initialize the high-pass filter with specified characteristics.
        
        Args:
            cutoff_freq (float): Cutoff frequency in Hz.
            order (int): Filter order, defaults to 1.
            q_factor (float): Quality factor, defaults to 1 (not used in Butterworth).
            insertion_losses_dB (float): Insertion losses in dB, defaults to 0.
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
        """
        self.temp_kelvin = temp_kelvin
        self.cutoff_freq = cutoff_freq
        self.order = order
        self.q_factor = q_factor
        self.insertion_losses_dB = insertion_losses_dB
        self.gain_losses = gain_db_to_gain(-self.insertion_losses_dB)  # Convert insertion loss to linear gain

    def get_rf_parameters_adapted_to_signals(self, signals: Signals, temp_kelvin: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get frequency-dependent gains and noise figures for the high-pass filter.
        
        Args:
            signals (Signals): Input signals object.
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to instance temperature.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Frequencies, gains, noise figures.
        """
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin

        freqs = signals.freqs[signals.freqs >= 0]

        # Design Butterworth high-pass filter
        b, a = butter(self.order, self.cutoff_freq, btype='high', fs=signals.sampling_rate, output='ba')
        w, h = freqz(b, a, worN=freqs, fs=signals.sampling_rate)

        gains    = h * self.gain_losses  # Apply insertion loss
        gains_db = np.minimum(0., gain_to_gain_db(gains))  # Cap at 0 dB (passive device)
        nf__s    = nf_db_to_nf(-gains_db)  # Noise figure based on loss

        return freqs, gains, nf__s, infs_like(freqs), infs_like(freqs), infs_like(freqs)  # No op1db, iip3, iip2 for passive device

# ====================================================================================================
# Antenna Component Class
# ====================================================================================================

class Antenna_Component(RF_Modelised_Component):
    """Class representing an antenna with frequency-dependent gain and phase.
    
    Attributes:
        temp_kelvin (float): Temperature in Kelvin.
        freqs (np.ndarray): Frequency points in Hz (extended with out-of-band).
        gains_db (np.ndarray): Gains in dB at specified frequencies.
        gains (np.ndarray): Linear gains (complex if phases provided, extended).
        phases_rad (Optional[np.ndarray]): Phases in radians at specified frequencies.
        freqs_sup (np.ndarray): Supplementary frequencies for out-of-band (high).
        freqs_inf (np.ndarray): Inferior frequencies for out-of-band (low).
        gains_sup_db (np.ndarray): Supplementary gains in dB for out-of-band (high).
        gains_sup (np.ndarray): Supplementary linear gains for out-of-band (high).
        gains_inf (np.ndarray): Inferior linear gains for out-of-band (low).
    """

    def __init__(self, freqs: np.ndarray, gains_db: np.ndarray, phases_rad: Optional[np.ndarray] = None, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        """Initialize the antenna with specified gain and phase characteristics.
        
        Args:
            freqs (np.ndarray): Frequency points in Hz.
            gains_db (np.ndarray): Gains in dB at each frequency.
            phases_rad (Optional[np.ndarray]): Phases in radians, defaults to None.
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
        """
        super().__init__(freqs, gains_db, phases_rad, temp_kelvin=temp_kelvin)

    def process_signals(self, signals: Signals, temp_kelvin: Optional[float] = None) -> None:
        """Process signals by applying antenna gains and adding thermal noise.
        
        Args:
            signals (Signals): Input signals object.
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to instance temperature.
        """
        temp_kelvin = temp_kelvin if temp_kelvin is not None else self.temp_kelvin
        super().process_signals(signals, temp_kelvin)

        signals.sig2d += Signals.generate_noise_dbm(signals.shape, thermal_noise_power_dbm(temp_kelvin, signals.bw_hz))  # Add thermal noise

# ====================================================================================================
# Main Execution
# ====================================================================================================

def main() -> None:
    """Main function to demonstrate the usage of the RF modeling classes."""
    # Example usage of the classes
    # Create a signal with noise and tones
    signal = Signals(20e9, 100e6)  # fmax = 40 GHz, bin_width = 10 MHz (duration = 1/bin_width = 100 ns)
    signal.add_noise(thermal_noise_power_dbm(signal.temp_kelvin, signal.bw_hz))  # Add thermal noise
    signal.add_tone(3e9, 0, 0)          # Add tone at 3 GHz, 0 dBm
    signal.add_tone(11e9, -55, pi / 4)  # Add tone at 11 GHz, -55 dBm
    signal.add_tone(17e9, -50, -pi / 3) # Add tone at 17 GHz, -50 dBm

    # Print RMS value
    print(f"Initial RMS value: {signal.rms_dbm()} dBm")

    # Plot temporal signal
    signal.plot_temporal(tmax=10e-9)

    # Plot spectrum
    signal.plot_spectrum()

    # Create an amplifier with non-linearities
    amplifier = Simple_Amplifier(gain_db=20, iip2_dbm=30, oip3_dbm=20, nf_db=5)

    # Process the signal through the amplifier
    amplifier.process(signal)

    # Print RMS value after amplification
    print(f"RMS value after amplification: {signal.rms_dbm()} dBm")

    # Plot temporal signal after amplification
    signal.plot_temporal(tmax=10e-9)

    # Plot spectrum after amplification
    signal.plot_spectrum()

    # Create a high-pass filter
    high_pass_filter = HighPassFilter(cutoff_freq=6e9, order=5, q_factor=0.7)

    # Apply the high-pass filter to the amplified signal
    high_pass_filter.process(signal)

    # Print RMS value after filtering
    print(f"RMS value after filtering: {signal.rms_dbm()} dBm")

    # Plot temporal signal after filtering
    signal.plot_temporal(tmax=10e-9)

    # Plot spectrum after filtering
    signal.plot_spectrum()

    # Testing components
    components = [
        Antenna_Component(freqs=(1e9, 20e9), gains_db=[3,7]),
        HighPassFilter(cutoff_freq=6e9, order=5, q_factor=0.7),
        Simple_Amplifier(gain_db=15, iip2_dbm=30, oip3_dbm=20, nf_db=5),
        Attenuator(att_db=5),
        Simple_Amplifier(gain_db=15, iip2_dbm=30, oip3_dbm=20, nf_db=5),
        RF_Cable(length_m=10, insertion_losses_dB=0.5),
    ]

    chain = RF_chain(components)

    for component in components+[chain]:
        print(f"\n=========== Assessing {component.__class__.__name__}")
        
        freqs, gains, phases, nf = component.assess_gain()
        plt.figure()
        plt.plot(freqs / 1e9, gains)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Gain (dB)')
        plt.title(f'Gain vs Frequency for {component.__class__.__name__}')
        
        results = component.assess_ipx(freq_min=1e9, freq_max=10e9, fr_stp=1e9)
        for freq, (gain_db, op1db_dbm, iip2_dbm, oip3_dbm) in results:
            print(f"Frequency: {freq / 1e9} GHz, Gain: {gain_db} dB, OP1dB: {op1db_dbm} dBm, IIP2: {iip2_dbm} dBm, OIP3: {oip3_dbm} dBm")
    
    plt.show()  # Display all plots

if __name__ == '__main__':
    main()
# ====================================================================================================
