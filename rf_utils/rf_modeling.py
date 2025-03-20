#!/usr/bin/env python
# coding: utf-8

"""
Project: RF_chain_modeling
GitHub: https://github.com/dunaar/RF_chain_modeling
Auteur: Pessel Arnaud
"""

# ====================================================================================================
# RF Signal Simulation and Analysis Framework
#
# This script provides a comprehensive framework for simulating and analyzing RF (Radio Frequency)
# signals and components. It includes classes and functions for signal generation, processing,
# and visualization.
#
# Key Features:
# - Signal generation with tones and noise.
# - RF component modeling (attenuators, amplifiers, cables, filters, antennas).
# - Signal processing and analysis.
# - Visualization of temporal and spectral representations.
#
# Author: Pessel Arnaud
# Date: 2025-03-15
# ====================================================================================================

import copy
from tqdm import tqdm
import itertools

import numpy as np
from numpy import pi

from scipy.signal import butter, sosfilt, freqz
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

# Constants
k_B = 1.38e-23  # Boltzmann constant in Joules per Kelvin
temperature_default = 298.15  # Default temperature in Kelvin: 298,15K = 25°C

# ====================================================================================================
# Utility Functions
# ====================================================================================================

def dbm_to_watts(power_dbm):
    """Convert power from dBm to watts."""
    return 10 ** (power_dbm / 10) / 1000

def watts_to_voltage(power_watts, imped_ohms=50):
    """Convert power from watts to voltage."""
    return np.sqrt(power_watts * imped_ohms)

def dbm_to_voltage(power_dbm, imped_ohms=50):
    """Convert power from dBm to voltage."""
    return watts_to_voltage(dbm_to_watts(power_dbm), imped_ohms)

def gain_db_to_gain(gain_db):
    """Convert gain from dB to linear scale."""
    return 10 ** (gain_db / 20.)

def gain_to_gain_db(gain):
    """Convert gain from linear scale to dB."""
    return 20 * np.log10(np.abs(gain))

def nf_db_to_nf(nf_db):
    """Convert noise figure from dB to linear scale."""
    return np.sqrt(10 ** (nf_db / 10) - 1)

def nf_to_nf_db(nf):
    """Convert noise figure from linear scale to dB."""
    return 10 * np.log10(nf ** 2 + 1)

def mul_nfs(nf1, nf2):
    """Multiply noise factors."""
    return np.sqrt((nf1 ** 2 + 1) * (nf2 ** 2 + 1) - 1)

def voltage_to_watts(voltage, imped_ohms=50):
    """Convert voltage to power in watts."""
    return voltage ** 2 / imped_ohms

def watts_to_dbm(power_watts):
    """Convert power from watts to dBm."""
    return 10 * np.log10(power_watts * 1000)

def voltage_to_dbm(voltage, imped_ohms=50):
    """Convert voltage to power in dBm."""
    return watts_to_dbm(voltage_to_watts(voltage, imped_ohms))

def calculate_rms(signal):
    """Calculate the RMS value of a signal."""
    return np.sqrt(np.mean(np.abs(signal) ** 2))

def calculate_rms_dbm(signal):
    """Calculate the RMS value of a signal in dBm."""
    return voltage_to_dbm(calculate_rms(signal))

def thermal_noise_power_dbm(temp_kelvin, bw_hz):
    """Calculate thermal noise power in dBm."""
    return watts_to_dbm(k_B * temp_kelvin * bw_hz)

def compute_spectrums(sigxd, sampling_rate):
    """Compute the frequency spectrum of a signal."""
    if len(sigxd.shape) == 1:
        sig2d = sigxd.reshape(1, sigxd.shape[0])
    else:
        sig2d = sigxd

    n_windows = sig2d.shape[0]
    n_points = sig2d.shape[1]

    freqs = np.fft.fftfreq(n_points, 1 / sampling_rate)
    spectrums = np.fft.fft(sig2d, axis=1) / n_points

    return freqs, spectrums

def get_spectrums_power_n_phase(freqs, spectrums, n_windows=1, imped_ohms=50):
    """Compute the power and phase spectrum of a signal."""
    spects_amp = abs(spectrums)
    spects_power = voltage_to_dbm(spects_amp, imped_ohms)
    spects_phase = np.angle(spectrums)  # Calculate the phase
    return spects_power, spects_phase

def plot_temporal_signal(time, sigxd, tmax=None, tmin=None):
    """Plot the temporal representation of a signal."""
    time = np.array(time)

    tmin = tmin if tmin else 0
    tmax = tmax if tmax else time.max()

    idx_min = np.argmin(np.abs(time - tmin))
    idx_max = np.argmin(np.abs(time - tmax))

    if (time[-1] - time[0]) > 10:
        unit = 's'
    elif (time[-1] - time[0]) > 10e-3:
        time = time * 1e3
        unit = 'ms'
    elif (time[-1] - time[0]) > 10e-6:
        time = time * 1e6
        unit = 'us'
    else:
        time = time * 1e9
        unit = 'ns'

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.subplots(1, 1, sharex=True)

    if len(sigxd.shape) == 1:
        ax1.plot(time[idx_min:idx_max], sigxd[idx_min:idx_max])
    elif len(sigxd.shape) == 2:
        for idx in range(sigxd.shape[0]):
            ax1.plot(time[idx_min:idx_max], sigxd[idx, idx_min:idx_max])

    ax1.set_xlabel(f'Time ({unit})')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    plt.tight_layout()

    
def plot_signal_spectrum(freqs, spectrum_power, spectrum_phase=None,
                         title_power="Spectre de puissance", title_phase="Spectre de phase",
                         ylabel_power="Puissance (dBm)"    , ylabel_phase="Phase (radians)"):
    """
    Plot the frequency and phase spectrum of a signal with customizable title and y-axis labels.

    Parameters:
    -----------
    freqs : array-like
        Frequency values for the x-axis.
    spectrum_power : array-like
        Power spectrum values for the y-axis.
    spectrum_phase : array-like, optional
        Phase spectrum values for the y-axis. Default is None.
    title : str, optional
        Title of the plot. Default is "Spectre de Fréquence".
    ylabel_power : str, optional
        Label for the y-axis of the power spectrum plot. Default is "Puissance (dBm)".
    ylabel_phase : str, optional
        Label for the y-axis of the phase spectrum plot. Default is "Phase (radians)".
    """
    freqs = np.array(freqs)
    idx_max = np.argmax(freqs) + 1

    unit = ''
    if freqs.max() - freqs[0] > 10e9:
        freqs = freqs / 1e9
        unit = 'GHz'
    elif freqs.max() - freqs[0] > 10e6:
        freqs = np.array(freqs) / 1e6
        unit = 'MHz'
    elif freqs.max() - freqs[0] > 10e3:
        freqs = np.array(freqs) / 1e3
        unit = 'kHz'
    else:
        unit = 'Hz'

    fig = plt.figure(figsize=(12, 6))

    if spectrum_phase is not None:
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
    else:
        ax1 = fig.subplots(1, 1)
        ax2 = None

    if len(spectrum_power.shape) == 1:
        ax1.plot(freqs[:idx_max], spectrum_power[:idx_max])
        if spectrum_phase is not None:
            ax2.plot(freqs[:idx_max], spectrum_phase[:idx_max])
    elif len(spectrum_power.shape) == 2:
        for idx in range(spectrum_power.shape[0]):
            ax1.plot(freqs[:idx_max], spectrum_power[idx][:idx_max])
            if spectrum_phase is not None:
                ax2.plot(freqs[:idx_max], spectrum_phase[idx][:idx_max])

    ax1.set_ylim(spectrum_power.min() - 1., spectrum_power.max() + 1.)
    ax1.set_xlabel(f'Fréquence ({unit})')
    ax1.set_ylabel(ylabel_power)
    ax1.set_title(title_power)
    ax1.grid(True)

    if spectrum_phase is not None:
        ax2.set_ylim(-pi, pi)
        ax2.set_xlabel(f'Fréquence ({unit}Hz)')
        ax2.set_ylabel(ylabel_phase)
        ax2.set_title(title_phase)
        ax2.grid(True)

    plt.tight_layout()

# ====================================================================================================
# Signals Class
# ====================================================================================================

class Signals:
    """
    Class for managing multiple microwave signals.

    Attributes:
    -----------
    fmax : float
        Maximum frequency of signals.
    bin_width : float
        Width of the spectral bins.
    n_windows : int
        Number of signal windows (default=32).
    imped_ohms : float
        Impedance, default=50.
    temp_kelvin : float
        Temperature in Kelvin, default=temperature_default.
    """

    @staticmethod
    def generate_signal_dbm(time, freq, power_dbm, phase, imped_ohms=50):
        """Generate a monotone signal."""
        amp_rms = dbm_to_voltage(power_dbm, imped_ohms)
        amp_pk = np.sqrt(2) * amp_rms
        return amp_pk * np.sin(2 * pi * freq * time + phase)

    @staticmethod
    def generate_noise_dbm(shape, power_dbm, imped_ohms=50):
        """Generate noise with the specified power in dBm.
           shape may be a tuple or a number."""
        amp = dbm_to_voltage(power_dbm, imped_ohms)
        return np.random.normal(0, amp, shape)

    def __init__(self, fmax, bin_width, n_windows=32, imped_ohms=50, temp_kelvin=temperature_default):
        """Initialize the Signals object."""
        self.fmax = fmax
        self.bin_width = bin_width
        self.imped_ohms = imped_ohms
        self.temp_kelvin = temp_kelvin
        self.n_windows = n_windows

        self.duration = 1 / bin_width
        self.n_points = int(np.ceil(2.2 * fmax / bin_width))
        self.sampling_rate = self.n_points / self.duration
        self.freqs = np.fft.fftfreq(self.n_points, 1 / self.sampling_rate)

        self.bw_hz = self.sampling_rate / 2

        self.shape = (self.n_windows, self.n_points)
        self.size = np.prod(self.n_windows * self.n_points)

        self.time = np.linspace(0, self.duration, self.n_points, endpoint=False)
        self.sig2d = Signals.generate_noise_dbm(self.shape, -1000)  # Very low noise to avoid log errors

        self.spectrum_uptodate = False

    def __getitem__(self, key):
        if key in ('freqs', 'spectrums', 'spects_power', 'spects_phase'):
            self.compute_spectrum()
            return getattr(self, '_' + key)

    def compute_spectrum(self, force=False):
        """Compute the frequency spectrum of the signals."""
        if force or not self.spectrum_uptodate:
            self._freqs, self._spectrums = compute_spectrums(self.sig2d, self.sampling_rate)
            self._spects_power, self._spects_phase = get_spectrums_power_n_phase(self._freqs, self._spectrums, self.imped_ohms)
            self.spectrum_uptodate = True

    def get_arg_freq(self, freq):
        """Get the index of the closest frequency in the spectrum."""
        return np.argmin(np.abs(self['freqs'] - freq))

    def add_signal(self, sigxd):
        """Add a signal to the existing signals."""
        if sigxd.shape == self.sig2d.shape:
            self.sig2d += sigxd
        elif len(sigxd.shape) == 2 and sigxd.shape[0] == 1 and sigxd.shape[1] == self.sig2d.shape[1]:
            self.sig2d[:] += sigxd[0]
        elif len(sigxd.shape) == 1 and sigxd.shape[0] == self.sig2d.shape[1]:
            self.sig2d[:] += sigxd
        else:
            raise ValueError(f"Invalid input, sigxd.shape: {sigxd.shape}")

        self.spectrum_uptodate = False

    def add_tone(self, freq, power_dbm, phase):
        """Add a tone to the signals."""
        self.add_signal(Signals.generate_signal_dbm(self.time, freq, power_dbm, phase, self.imped_ohms))

    def add_noise(self, power_dbm):
        """Add noise to the signals."""
        self.add_signal(Signals.generate_noise_dbm(self.shape, power_dbm))

    def add_thermal_noise(self, temp_kelvin=None):
        """Add thermal noise to the signals."""
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin
        self.add_noise(thermal_noise_power_dbm(temp_kelvin, self.bw_hz))

    def rms_dbm(self):
        """Calculate the RMS value of the signals in dBm."""
        return calculate_rms_dbm(self.sig2d)

    def plot_temporal(self, tmax=None, tmin=None):
        """Plot the temporal representation of the signals."""
        idx_min = int((tmin if tmin else 0) * self.sampling_rate)
        idx_max = int((tmax if tmax else self.duration) * self.sampling_rate)

        plot_temporal_signal(self.time[idx_min:idx_max], self.sig2d[:, idx_min:idx_max])

    def plot_spectrum(self, fmax=None, fmin=None):
        """Plot the frequency spectrum of the signals."""
        self.compute_spectrum()
        plot_signal_spectrum(self['freqs'], self['spects_power'], self['spects_phase'])

# ====================================================================================================
# RF Component Base Class
# ====================================================================================================

class RF_Component(object):
    """Base class for RF components."""
    ft = lambda self, x,k: np.tanh(x/k)*k

    def __init__(self):
        pass

    def process_signals(self, signals, temp_kelvin=None):
        """Generic method to process signals. Should be overridden by subclasses."""
        return None

    def process(self, signals, temp_kelvin=None, inplace=True):
        """Process signals with the RF component."""
        if not inplace:
            signals = copy.deepcopy(signals)

        means = (1.-1e-3)*signals.sig2d.mean(1)
        signals.sig2d = signals.sig2d - means[:, np.newaxis] # DC block: continuous signal is removed
        
        self.process_signals(signals, temp_kelvin=temp_kelvin)

        signals.spectrum_uptodate = False

        return signals if not inplace else None

    def assess_gain(self, fmin=400e6, fmax=19e9, step=100e6, temp_kelvin=temperature_default):
        """Assess the gain and phase versus frequency of the RF component."""
        print("""\nAssess the gain and phase versus frequency of the RF component.""")
        
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

            gains[idx_frq] = clear_signal['spects_power'][0, arg_frq_pos]
            phass[idx_frq] = clear_signal['spects_phase'][0, arg_frq_pos]

            proc_clear_signal = self.process(clear_signal, temp_kelvin=temp_kelvin, inplace=False)

            gains[idx_frq] = proc_clear_signal['spects_power'][0, arg_frq_pos] - gains[idx_frq]
            phass[idx_frq] = proc_clear_signal['spects_phase'][0, arg_frq_pos] - phass[idx_frq]

            # Compute noise figure
            signals = copy.deepcopy(noisy_signals)
            signals.add_signal(clear_signal.sig2d)

            spects_befor = (signals['spectrums'][:, arg_frq_neg] +
                            np.conj(signals['spectrums'][:, arg_frq_pos])) / 2

            self.process(signals, temp_kelvin=temp_kelvin)

            spects_after = (signals['spectrums'][:, arg_frq_neg] +
                            np.conj(signals['spectrums'][:, arg_frq_pos])) / 2

            pwr_sig_befor = voltage_to_dbm(np.abs(spects_befor.mean()))
            pwr_sig_after = voltage_to_dbm(np.abs(spects_after.mean()))

            pwr_nse_befor = voltage_to_dbm(np.sqrt(spects_befor.var()))
            pwr_nse_after = voltage_to_dbm(np.sqrt(spects_after.var()))

            n_fgs[idx_frq] = (pwr_sig_befor - pwr_nse_befor) - (pwr_sig_after - pwr_nse_after)

        phass[phass < -pi] += 2 * pi
        phass[phass > pi] -= 2 * pi

        #for var in ('freqs_for_test', 'gains', 'phass, n_fgs'):
        #    print('%s: '%(var), eval(var))

        return freqs_for_test, gains, phass, n_fgs

    def assess_iipx(self, fc=9e9, df=400e6, temp_kelvin=temperature_default):
        """Assess the IP2 and IP3 (Intercept Point of order 2 and 3) of the RF component."""
        print("""\nAssess the IP2 and IP3 (Intercept Point of order 2 and 3) of the RF component.""")
        
        #--------------------------------------------------------
        fmax = 2.5 * fc
        bin_width = df / 32
        n_windows = 8
        #--------------------------------------------------------

        #--------------------------------------------------------
        input_pwr_m, outpt_pwr_m = [], []
        input_pwr_d, outpt_pwr_d, im2___power, im3___power = [], [], [], []
    
        gains_db, iip2s_dbm, iip3s_dbm = [], [], []
        ip1db_dbm, op1db_dbm = None, None
        #--------------------------------------------------------

        #--------------------------------------------------------
        f1 = fc - df
        f2 = fc + df
        f1pf2 = f1 + f2
        df1mf2 = 2 * f1 - f2
        df2mf1 = 2 * f2 - f1
        
        # Indexes of frequencies in spectrum
        signals = Signals(fmax, bin_width, n_windows=n_windows, imped_ohms=50, temp_kelvin=temp_kelvin)

        arg_frq_fc = signals.get_arg_freq(fc)
        arg_frq_f1 = signals.get_arg_freq(f1)
        arg_frq_f2 = signals.get_arg_freq(f2)
        arg_frq_f1pf2 = signals.get_arg_freq(f1pf2)
        arg_frq_df1mf2 = signals.get_arg_freq(df1mf2)
        arg_frq_df2mf1 = signals.get_arg_freq(df2mf1)
        #--------------------------------------------------------

        #--------------------------------------------------------
        for power_dbm in tqdm(range(-50, 50+1), desc="Assessing Gain"):
            # Initialize monotone signals
            signals = Signals(fmax, bin_width, n_windows=n_windows, imped_ohms=50, temp_kelvin=temp_kelvin)
            signals.add_tone(fc, power_dbm, 0)
            
            input_pwr_m.append( np.mean(signals['spects_power'][:, arg_frq_fc]) )

            # Processing signal
            self.process(signals, temp_kelvin=temp_kelvin)
    
            # Append values in tables
            outpt_pwr_m.append( np.mean(signals['spects_power'][:, arg_frq_fc]) )
            
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
        #--------------------------------------------------------
    
        #--------------------------------------------------------
        for power_dbm in tqdm(range(-50, 50+1), desc="Assessing IP2, IP3"):
            # Initialize bitone signals
            signals = Signals(fmax, bin_width, n_windows=n_windows, imped_ohms=50, temp_kelvin=temp_kelvin)
            signals.add_tone(f1, power_dbm, 0)
            signals.add_tone(f2, power_dbm, 0)

            input_pwr_d.append(np.mean((signals['spects_power'][:, arg_frq_f1] +
                                        signals['spects_power'][:, arg_frq_f2]) / 2))

            # Processing signal
            self.process(signals, temp_kelvin=temp_kelvin)

            # Append values in tables
            outpt_pwr_d.append(np.mean((signals['spects_power'][:, arg_frq_f1] +
                                        signals['spects_power'][:, arg_frq_f2]) / 2))

            im2___power.append(np.mean(signals['spects_power'][:, arg_frq_f1pf2]))

            im3___power.append(np.mean((signals['spects_power'][:, arg_frq_df1mf2] +
                                        signals['spects_power'][:, arg_frq_df2mf1]) / 2))

            if len(input_pwr_d) >= 2:
                delta_input_pwr_d = input_pwr_d[-1] - input_pwr_d[-2]
                delta_im2___power = im2___power[-1] - im2___power[-2]
                delta_im3___power = im3___power[-1] - im3___power[-2]
                
                if outpt_pwr_d[-1] < op1db_dbm:
                    if np.abs(delta_im2___power / 2 - delta_input_power) / delta_input_power < 0.02:
                        iip2s_dbm.append( input_pwr_d[-1] + (outpt_pwr_d[-1] - im2___power[-1]) )
        
                    if np.abs(delta_im3___power / 3 - delta_input_power) / delta_input_power < 0.02:
                        iip3s_dbm.append( input_pwr_d[-1] + (outpt_pwr_d[-1] - im3___power[-1]) / 2 )

        idx_mid = len(outpt_pwr_d)//2
        if iip2s_dbm:
            iip2_dbm = np.array(iip2s_dbm).mean()
        else:
            print("Error: Unable to characterize IP2 finely.")
            iip2_dbm = input_pwr_d[idx_mid] + (outpt_pwr_d[idx_mid] - im2___power[idx_mid])
        oip2_dbm  = iip2_dbm + gain_db
        
        if iip3s_dbm:
            iip3_dbm = np.array(iip3s_dbm).mean()
        else:
            print("Error: Unable to characterize IP3 finely.")
            if op1db_dbm:
                iip3_dbm  = ip1db_dbm + 10.
            else:
                iip3_dbm = input_pwr_d[idx_mid] + (outpt_pwr_d[idx_mid] - im3___power[idx_mid]) / 2
        oip3_dbm  = iip3_dbm + gain_db
        #--------------------------------------------------------

        #--------------------------------------------------------
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
        #--------------------------------------------------------
    
        for var in ('gain_db', 'op1db_dbm', 'iip2_dbm', 'oip3_dbm'):
            print('%s: '%(var), eval(var))
        
        return gain_db, op1db_dbm, iip2_dbm, oip3_dbm

# ====================================================================================================
# RF Channel Class
# ====================================================================================================

class RF_chain(RF_Component):
    """Class representing an RF chain composed of multiple RF components."""

    def __init__(self, rf_components=[]):
        self.rf_components = list(rf_components)

    def process_signals(self, signals, temp_kelvin=None):
        """Process signals through each RF component in the channel."""
        for rf_compnt in self.rf_components:
            rf_compnt.process(signals, temp_kelvin)

# ====================================================================================================
# Attenuator Class
# ====================================================================================================

class Attenuator(RF_Component):
    """Class representing an attenuator RF component."""

    def __init__(self, att_db, temp_kelvin=temperature_default):
        """Initialize the attenuator with a specified attenuation in dB."""
        self.temp_kelvin = temp_kelvin
        self.att_db = att_db
        self.nf_db = att_db

        self.gain = gain_db_to_gain(-att_db)  # Convert gain from dB to linear scale
        self.nf = nf_db_to_nf(self.nf_db)

    def process_signals(self, signals, temp_kelvin=None):
        """Process signals by applying attenuation and adding noise."""
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin

        signals.sig2d += self.nf * Signals.generate_noise_dbm(signals.shape, thermal_noise_power_dbm(temp_kelvin, signals.bw_hz))
        signals.sig2d *= self.gain

# ====================================================================================================
# Simple Amplifier Class
# ====================================================================================================

class Simple_Amplifier(RF_Component):
    """Class representing a simple amplifier with gain, noise figure, and non-linearities."""

    def __init__(self, gain_db, nf_db, op1db_dbm=20, oip3_dbm=10, iip2_dbm=40, temp_kelvin=temperature_default):
        """Initialize the amplifier with specified gain, noise figure, and intercept points."""
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
        self.a2     = -0.43*self.a1/self.iip2
        self.k_oip3 = 6.5e-1*self.oip3
    
    ft = lambda self, x,k: np.tanh(x/k)*k
    def process_signals(self, signals, temp_kelvin=None):
        """Process signals by applying gain, adding noise, and introducing non-linearities."""
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin

        signals.sig2d += self.nf * Signals.generate_noise_dbm(signals.shape, thermal_noise_power_dbm(temp_kelvin, signals.bw_hz))

        s   = signals.sig2d
        
        s1  = self.ft(self.a1*s, self.k_oip3)
        
        s2  = self.a2*s**2
        s2 -= s2.mean(1)[:, np.newaxis]
        s2  = self.ft(s2, self.op1db*1.2)
        
        # OP1dB: 1dB compression point output power
        signals.sig2d = self.ft(s1+s2, self.op1db*6)
        
        #signals.sig2d = self.ft(self.a1 * signals.sig2d, self.k_oip3) + self.a2 * self.ft(signals.sig2d, self.op1db * 2) ** 2
        # OP1dB: 1dB compression point output power
        #signals.sig2d = self.ft(signals.sig2d, self.op1db * 6)

# ====================================================================================================
# RF Modelised Component Class
# ====================================================================================================

class RF_modelised_component(RF_Component):
    """Class representing a modelised RF component with specified gain, noise figure, and phase characteristics."""

    def __init__(self, freqs, gains_db, nfs_db, phases_rad=None, nominal_gain_for_im_db=None, op1db_dbm=np.inf, oip3_dbm=np.inf, iip2_dbm=np.inf, temp_kelvin=temperature_default):
        """Initialize the modelised component with specified characteristics."""
        self.temp_kelvin = temp_kelvin

        if freqs is None or gains_db is None or nfs_db is None:
            self.freqs = None
            self.gains_db = None
            self.gains = None
            self.nfs_db = None
            self.nfs = None
            self.phases_rad = None
        else:
            self.freqs = np.array(freqs)
            self.gains_db = np.array(gains_db)
            self.gains = gain_db_to_gain(self.gains_db)
            self.nfs_db = np.array(nfs_db)
            self.nfs = nf_db_to_nf(self.nfs_db)

            if phases_rad is not None:
                self.phases_rad = np.array(phases_rad)
                self.gains = self.gains * np.exp(1j * self.phases_rad)

        if nominal_gain_for_im_db is not None and nominal_gain_for_im_db > 0:
            self.nominal_gain_for_im_db = nominal_gain_for_im_db
            self.op1db_dbm = op1db_dbm
            self.oip3_dbm = oip3_dbm
            if iip2_dbm is not np.inf:
                self.iip2_dbm = iip2_dbm
            elif oip3_dbm is not np.inf:
                self.iip2_dbm = 40.
            else:
                self.iip2_dbm = np.inf
        elif nominal_gain_for_im_db is not None:
            print('[RF_modelised_component] Error: nominal_gain_for_im_db:', nominal_gain_for_im_db)
            nominal_gain_for_im_db = None

        if nominal_gain_for_im_db is None:
            self.nominal_gain_for_im_db = 1e-99
            self.op1db_dbm = np.inf
            self.oip3_dbm = np.inf
            self.iip2_dbm = np.inf

        self.iip2 = dbm_to_voltage(self.iip2_dbm)
        self.oip3 = dbm_to_voltage(self.oip3_dbm)
        self.op1db = dbm_to_voltage(self.op1db_dbm)

        # For distortion
        self.a1 = gain_db_to_gain(self.nominal_gain_for_im_db)
        self.a2 = 0.5 * self.a1 / self.iip2
        self.k_oip3 = 0.24 * 10 ** (self.oip3_dbm / 20)

        # Initializing parameters to manage gains and losses out of band
        self.freqs_sup = 100e6 * np.arange(1, 11)
        self.freqs_inf = -self.freqs_sup[::-1]

        self.gains_sup_db = -5. * np.arange(1, 11)
        self.gains_sup = gain_db_to_gain(self.gains_sup_db)
        self.gains_inf = self.gains_sup[::-1]

        self.nfs_sup = nf_db_to_nf(-self.gains_sup_db)
        self.nfs_inf = self.nfs_sup[::-1]

    def get_gains_nfs(self, signals, temp_kelvin=None):
        """Get gains and noise figures for the signals."""
        return self.freqs, self.gains, self.nfs

    def process_signals(self, signals, temp_kelvin=None):
        """Process signals by applying gains, adding noise, and introducing distortion."""
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin

        # Get gains and noise figures
        freqs, gains, nfs = self.get_gains_nfs(signals, temp_kelvin)

        # Extend the frequency domain with losses and noise figures
        freqs = np.concatenate((self.freqs_inf + freqs[0], freqs, self.freqs_sup + freqs[-1]))

        gains_inf = self.gains_inf * np.exp(1j * np.linspace(-np.angle(gains[0]), 0, num=len(self.gains_inf), endpoint=False))
        gains_sup = self.gains_sup * np.exp(1j * np.linspace(-np.angle(gains[-1]), 0, num=len(self.gains_sup), endpoint=False)[::-1])

        gains = np.concatenate((gains_inf * gains[0], gains, gains_sup * gains[-1]))
        nfs = np.concatenate((mul_nfs(self.nfs_inf, nfs[0]), nfs, mul_nfs(self.nfs_sup, nfs[-1])))

        # Extend to the negative frequencies
        gains = np.concatenate((np.conjugate(gains[freqs > 0][::-1]), gains[freqs >= 0]))
        nfs = np.concatenate((nfs[freqs > 0][::-1], nfs[freqs >= 0]))
        freqs = np.concatenate((-freqs[freqs > 0][::-1], freqs[freqs >= 0]))

        interp_gains = interp1d(freqs, gains, kind='linear', bounds_error=False, fill_value=(gains[0], gains[-1]))
        interp_nfs = interp1d(freqs, nfs, kind='linear', bounds_error=False, fill_value=(nfs[0], nfs[-1]))

        # Get spectrums of the signals
        spectrums = np.fft.fft(signals.sig2d, axis=1)
        fftfreqs = np.fft.fftfreq(signals.n_points, 1 / signals.sampling_rate)

        # Apply noise figures and gains
        spectrums += interp_nfs(fftfreqs) * np.fft.fft(Signals.generate_noise_dbm(signals.shape, thermal_noise_power_dbm(temp_kelvin, signals.bw_hz)), axis=1)
        spectrums *= interp_gains(fftfreqs)

        # Retrieve temporal signal
        signals.sig2d = np.real(np.fft.ifft(spectrums, axis=1))

        # Apply distortion
        if not self.oip3 > 1e308:
            signals.sig2d /= self.a1
            signals.sig2d = self.ft(self.a1 * signals.sig2d, self.k_oip3) + self.a2 * self.ft(signals.sig2d, self.op1db * 2) ** 2

        if not self.op1db > 1e308:
            signals.sig2d = self.ft(signals.sig2d, self.op1db * 11.5)

# ====================================================================================================
# RF Cable Class
# ====================================================================================================

class RF_Cable(RF_modelised_component):
    """Class representing an RF cable with insertion losses."""

    def __init__(self, length_m, alpha=5.2e-06, insertion_losses_dB=0, temp_kelvin=temperature_default):
        """Initialize the RF cable with specified length and insertion losses."""
        self.temp_kelvin = temp_kelvin
        self.length_m = length_m
        self.alpha = alpha
        self.insertion_losses_dB = insertion_losses_dB

        super().__init__(None, None, None, temp_kelvin=temp_kelvin)

    def get_gains_nfs(self, signals, temp_kelvin=None):
        """Get gains and noise figures for the RF cable."""
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin

        freqs = signals.freqs[signals.freqs > 0]
        gains_db = -self.insertion_losses_dB - self.alpha * np.sqrt(freqs) * self.length_m
        gains = gain_db_to_gain(gains_db)
        nfs = nf_db_to_nf(-gains_db)

        return freqs, gains, nfs

# ====================================================================================================
# High Pass Filter Class
# ====================================================================================================

class HighPassFilter(RF_modelised_component):
    """Class representing a high-pass filter with specified cutoff frequency and order."""

    def __init__(self, cutoff_freq, order=1, q_factor=1, insertion_losses_dB=0, temp_kelvin=temperature_default):
        """Initialize the high-pass filter with specified characteristics."""
        self.temp_kelvin = temp_kelvin
        self.cutoff_freq = cutoff_freq
        self.order = order
        self.q_factor = q_factor
        self.insertion_losses_dB = insertion_losses_dB
        self.gain_losses = 10 ** (-self.insertion_losses_dB / 20)  # Insertion losses as a negative gain

        super().__init__(None, None, None, temp_kelvin=temp_kelvin)

    def get_gains_nfs(self, signals, temp_kelvin=None):
        """Get gains and noise figures for the high-pass filter."""
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin

        freqs = signals.freqs[signals.freqs > 0]
        b, a = butter(self.order, self.cutoff_freq, btype='high', fs=signals.sampling_rate, output='ba')
        w, h = freqz(b, a, worN=freqs, fs=signals.sampling_rate)

        gains = h
        gains_db = np.minimum(0., gain_to_gain_db(gains))
        nfs = nf_db_to_nf(-gains_db)

        return freqs, gains, nfs

# ====================================================================================================
# Antenna Component Class
# ====================================================================================================

class Antenna_Component(RF_Component):
    """Class representing an antenna with specified gain and phase characteristics."""

    def __init__(self, freqs, gains_db, phases_rad=None, temp_kelvin=temperature_default):
        """Initialize the antenna with specified characteristics."""
        self.temp_kelvin = temp_kelvin

        self.freqs    = np.array(freqs)
        self.gains_db = np.array(gains_db)
        self.gains    = gain_db_to_gain(self.gains_db)
        
        if phases_rad is not None:
            self.phases_rad = np.array(phases_rad)
            self.gains = self.gains * np.exp(1j * self.phases_rad)

        # Initializing parameters to manage gains and losses out of band
        self.freqs_sup = 100e6 * np.arange(1, 11)
        self.freqs_inf = -self.freqs_sup[::-1]

        self.gains_sup_db = -5. * np.arange(1, 11)
        self.gains_sup    = gain_db_to_gain(self.gains_sup_db)
        self.gains_inf    = self.gains_sup[::-1]

        # Extend the frequency domain with losses and noise figures
        self.freqs = np.concatenate((self.freqs_inf + self.freqs[0], self.freqs, self.freqs_sup + self.freqs[-1]))

        self.gains_inf = self.gains_inf * np.exp(1j * np.linspace(-np.angle(self.gains[ 0]), 0, num=len(self.gains_inf), endpoint=False))
        self.gains_sup = self.gains_sup * np.exp(1j * np.linspace(-np.angle(self.gains[-1]), 0, num=len(self.gains_sup), endpoint=False)[::-1])

        self.gains = np.concatenate((self.gains_inf * self.gains[0], self.gains, self.gains_sup * self.gains[-1]))

        # Extend to the negative frequencies
        self.gains = np.concatenate((np.conjugate(self.gains[self.freqs > 0][::-1]), self.gains[self.freqs >= 0]))
        self.freqs = np.concatenate((-self.freqs[self.freqs > 0][::-1], self.freqs[self.freqs >= 0]))

    def process_signals(self, signals, temp_kelvin=None):
        """Process signals by applying antenna gains."""
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin

        interp_gains = interp1d(self.freqs, self.gains, kind='linear', bounds_error=False, fill_value=(self.gains[0], self.gains[-1]))

        # Get spectrums of the signals
        spectrums = np.fft.fft(signals.sig2d, axis=1)
        fftfreqs = np.fft.fftfreq(signals.n_points, 1 / signals.sampling_rate)

        spectrums *= interp_gains(fftfreqs)

        # Retrieve temporal signal
        signals.sig2d = np.real(np.fft.ifft(spectrums, axis=1))
        signals.sig2d += Signals.generate_noise_dbm(signals.shape, thermal_noise_power_dbm(temp_kelvin, signals.bw_hz))

# ====================================================================================================
# Main Execution
# ====================================================================================================

if __name__ == '__main__':
    # Create a signal with noise and tones
    signal = Signals(10e-9, 40e9)
    signal.add_noise(thermal_noise_power_dbm(signal.temp_kelvin, signal.bw_hz))
    signal.add_tone(3e9, 0, 0)
    signal.add_tone(11e9, -55, pi / 4)
    signal.add_tone(17e9, -50, -pi / 3)

    print(signal.rms_dbm())
    signal.plot_temporal(10e-9)
    signal.plot_spectrum()

    # Create an amplifier with non-linearities
    amplifier = Simple_Amplifier(gain_db=20, iip2_dbm=30, oip3_dbm=20, nf_db=5)

    # Process the signal through the amplifier
    signal.signal = amplifier.process(signal)

    print(signal.rms_dbm())
    signal.plot_temporal(10e-9)
    signal.plot_spectrum()

    # Create a high-pass filter
    high_pass_filter = HighPassFilter(cutoff_freq=6e9, order=5, q_factor=0.7)

    # Apply the high-pass filter to the amplified signal
    signal.signal = high_pass_filter.process(signal)

    print(signal.rms_dbm())
    signal.plot_temporal(10e-9)
    signal.plot_spectrum()
