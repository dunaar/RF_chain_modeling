#!/usr/bin/env python
# coding: utf-8

# ====================================================================================================
# RF Chain Modeling Utilities
# This module provides utilities for modeling an RF chain, including functions for unit conversion,
# signal processing, and RF component modeling.
# Author: Your Name
# Date: 2025-03-15
# ====================================================================================================

import copy
import numpy as np
from numpy import pi
from scipy.signal import butter, sosfilt, freqz
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Constants
k_B = 1.38e-23  # Boltzmann constant in Joules per Kelvin
temperature_default = 290  # Default temperature in Kelvin

# Unit conversion factors
D2R, R2D = np.pi/180., 180./np.pi
units = {
    'hz': (1., 'Hz'),
    'khz': (1e3, 'Hz'),
    'mhz': (1e6, 'Hz'),
    'ghz': (1e9, 'Hz'),
    'rad': (1., 'rad'),
    'deg': (D2R, 'rad'),
    'db': (1., 'dB'),
}

def convert_unit(unit):
    """Convert unit to a standard unit."""
    unit_factor = units.get(unit.lower(), None)
    if unit_factor is None:
        unit_factor = (1., unit)
    return unit_factor

def extrapolate_nan(freq, arr):
    """Extrapolate NaN values in an array based on frequency."""
    argsnan = np.where(np.isnan(arr))[0]
    if len(argsnan) > 0:
        argsnotnan = np.where(np.isnan(arr) == False)[0]
        arr[argsnan] = np.interp(freq[argsnan], freq[argsnotnan], arr[argsnotnan])

def read_from_csv(filepath, delim_field='\t', multi_df=False, delim_decim='.', delim_cmt='#'):
    """Read data from a CSV file."""
    comments = []
    dict_data = {'comments': comments, 'titles': None, 'units': [], 'data': None, 'length': 0}

    lines = []
    with open(filepath, 'r') as of:
        lines = of.readlines()

    length = 0
    for idx_line, line in enumerate(lines):
        try:
            # Keeping comments
            line = line.split(delim_cmt)
            if len(line) > 1:
                comments.append(delim_cmt.join(line[1:]))
            line = line[0]  # Keeping line except comments

            # Treating the fields
            if line.strip() == '':
                continue

            if multi_df:
                line = delim_field.join(s_ for s_ in line.split(delim_field) if s_ != '')

            fields = line.split(delim_field)

            if dict_data['titles'] is None:
                # No key in dict_data means fields are the column titles
                dict_data['titles'] = []

                for idx, field in enumerate(fields):
                    field = field.strip()
                    field = field.strip('"')
                    if field == '':
                        break

                    split = field.split('(')
                    title = split[0].strip().lower()
                    unit = split[1].split(')')[0].strip() if len(split) > 1 else ''

                    suffix = ''
                    while (title + suffix) in dict_data:
                        suffix = '_%d' % (eval(suffix[1:]) + 1) if suffix != '' else '_2'
                    title = title + suffix

                    dict_data['titles'].append(title)
                    dict_data[title] = {'unit': unit, 'data': []}

                dict_data['titles'] = tuple(dict_data['titles'])
                continue

            for idx, title in enumerate(dict_data['titles']):
                value = None

                if idx < len(fields):
                    try:
                        value = float(fields[idx].replace(delim_decim, '.'))
                    except:
                        print('Error! File <%s>, line %03d, field %02d: bad conversion' % (filepath, idx_line+1, idx+1))
                        print('%s' % (lines[idx_line]), end='')
                        print('<%s>' % (fields[idx]))
                        value = float('nan')
                else:
                    print('Error! File <%s>, line %03d, field %02d: missing values' % (filepath, idx_line+1, idx+1))
                    print(lines[idx_line])
                    value = float('nan')

                if value is None:
                    value = float('nan')
                    print(lines[idx_line])
                    print('Error! File <%s>, line %03d, field %02d: something was wrong' % (filepath, idx_line+1, idx+1))

                dict_data[title]['data'].append(value)
            length += 1
        except:
            print('%03d: <%s>' % (idx_line, line))
            raise

        dict_data['length'] = length

    dt = np.dtype([(title.lower(), 'float') for title in dict_data['titles']])
    dict_data['data'] = np.zeros(length, dtype=dt)

    for title in dict_data['titles']:
        factor, unit = convert_unit(dict_data[title]['unit'])
        dict_data[title]['unit'] = unit
        dict_data['units'].append(unit)
        dict_data['data'][title] = factor * np.array(dict_data[title]['data'])
        dict_data[title]['data'] = dict_data['data'][title]

    for title in dict_data['titles']:
        extrapolate_nan(dict_data['freq']['data'], dict_data[title]['data'])

    dict_data['units'] = tuple(dict_data['units'])

    return dict_data

def write_to_csv(filepath, dict_data, delim_field='\t', delim_decim='.', delim_cmt='#'):
    """Write data to a CSV file."""
    with open(filepath, 'w') as of:
        for cmt in dict_data['comments']:
            of.write('%s %s' % (delim_cmt, cmt))

        for idx_title, title in enumerate(dict_data['titles']):
            if idx_title > 0:
                of.write(delim_field)
            of.write('%s(%s)' % (title, dict_data['units'][idx_title]))
        of.write('\n')

        for idx_arr in range(dict_data['length']):
            for idx_title, title in enumerate(dict_data['titles']):
                if idx_title > 0:
                    of.write(delim_field)
                of.write('%s' % (dict_data[title]['data'][idx_arr]))
            of.write('\n')

# Conversion functions
def dbm_to_watts(power_dbm):
    """Convert dBm to watts."""
    return 10 ** (power_dbm / 10) / 1000

def watts_to_voltage(power_watts, imped_ohms=50):
    """Convert watts to voltage."""
    return np.sqrt(power_watts * imped_ohms)

def dbm_to_voltage(power_dbm, imped_ohms=50):
    """Convert dBm to voltage."""
    return watts_to_voltage(dbm_to_watts(power_dbm), imped_ohms)

def gain_db_to_gain(gain_db):
    """Convert gain in dB to linear gain."""
    return 10**(gain_db/20.)

def gain_to_gain_db(gain):
    """Convert linear gain to gain in dB."""
    return 20*np.log10(np.abs(gain))

def nf_db_to_nf(nf_db):
    """Convert noise figure in dB to linear noise factor."""
    return np.sqrt(10**(nf_db/10) - 1)

def nf_to_nf_db(nf):
    """Convert linear noise factor to noise figure in dB."""
    return 10*np.log10(nf**2 + 1)

def mul_nfs(nf1, nf2):
    """Multiply noise factors."""
    return np.sqrt((nf1**2 + 1) * (nf2**2 + 1) - 1)

def voltage_to_watts(voltage, imped_ohms=50):
    """Convert voltage to watts."""
    return voltage**2 / imped_ohms

def watts_to_dbm(power_watts):
    """Convert watts to dBm."""
    return 10 * np.log10(power_watts * 1000)

def voltage_to_dbm(voltage, imped_ohms=50):
    """Convert voltage to dBm."""
    return watts_to_dbm(voltage_to_watts(voltage, imped_ohms))

# Signal processing functions
def calculate_rms(signal):
    """Calculate the RMS value of a signal."""
    return np.sqrt(np.mean(np.abs(signal)**2))

def calculate_rms_dbm(signal):
    """Calculate the RMS value of a signal in dBm."""
    return voltage_to_dbm(calculate_rms(signal))

def thermal_noise_power_dbm(temp_kelvin, bw_hz):
    """Calculate the thermal noise power in dBm."""
    return watts_to_dbm(k_B * temp_kelvin * bw_hz)

def compute_spectrums(sigxd, sampling_rate):
    """Compute the frequency spectrums of a signal."""
    if len(sigxd.shape) == 1:
        sig2d = sigxd.reshape(1, signals.shape[0])
    else:
        sig2d = sigxd

    n_windows = sig2d.shape[0]
    n_points = sig2d.shape[1]

    freqs = np.fft.fftfreq(n_points, 1/sampling_rate)
    spectrums = np.fft.fft(sig2d, axis=1) / n_points

    return freqs, spectrums

def get_spectrums_power_n_phase(freqs, spectrums, n_windows=1, imped_ohms=50):
    """
    Calculate the power and phase spectrums of a signal in dBm and radians.

    Energy vs Power:
    The DFT calculates coefficients representing energy at each frequency.
    To obtain comparable channel power, normalize (divide) by the number of samples N.
    Without normalization, to obtain the RMS power of the spectrum, calculate the RMS as in the time domain:
    - root of the **mean** of squares.
    When the spectrum is normalized (already divided by N), the RMS power of the spectrum is:
    - root of the **sum** of squares.
    
    Interpretation of Frequency Components:
    The DFT of a real signal produces a symmetric spectrum. Negative and positive components must be correctly interpreted.
    For a real signal, energy is distributed between positive and negative frequencies.
    """
    spects_amp = abs(spectrums)
    spects_power = voltage_to_dbm(spects_amp, imped_ohms)
    spects_phase = np.angle(spectrums)  # Calculate the phase

    return spects_power, spects_phase

def plot_temporal_signal(time, sigxd, tmax=None, tmin=None):
    """Plot the temporal signal."""
    time = np.array(time)

    tmin = tmin if tmin else 0
    tmax = tmax if tmax else time.max()

    idx_min = np.argmin(np.abs(time-tmin))
    idx_max = np.argmin(np.abs(time-tmax))

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

    ax1.set_xlabel('Time (%s)' % (unit))
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    plt.tight_layout()

def plot_signal_spectrum(freqs, spectrum_power, spectrum_phase):
    """Plot the frequency spectrum and phase of a signal."""
    freqs = np.array(freqs)
    idx_max = np.argmax(freqs)+1

    unit = ''
    if freqs.max()-freqs[0] > 10e9:
        freqs = freqs / 1e9
        unit = 'GHz'
    elif freqs.max()-freqs[0] > 10e6:
        freqs = np.array(freqs) / 1e6
        unit = 'MHz'
    elif freqs.max()-freqs[0] > 10e3:
        freqs = np.array(freqs) / 1e3
        unit = 'kHz'
    else:
        unit = 'Hz'

    fig = plt.figure(figsize=(12, 6))
    ax1, ax2 = fig.subplots(2, 1, sharex=True)

    if len(spectrum_power.shape) == 1:
        ax1.plot(freqs[:idx_max], spectrum_power[:idx_max])
        ax2.plot(freqs[:idx_max], spectrum_phase[:idx_max])
    elif len(spectrum_power.shape) == 2:
        for idx in range(spectrum_power.shape[0]):
            ax1.plot(freqs[:idx_max], spectrum_power[idx][:idx_max])
            ax2.plot(freqs[:idx_max], spectrum_phase[idx][:idx_max])

    ax1.set_ylim(spectrum_power.min()-1., spectrum_power.max()+1.)
    ax1.set_xlabel('Frequency (%s)' % (unit))
    ax1.set_ylabel('Power (dBm)')
    ax1.set_title('Frequency Spectrum in dBm')
    ax1.grid(True)

    ax2.set_ylim(-pi, pi)
    ax2.set_xlabel('Frequency (%sHz)' % (unit))
    ax2.set_ylabel('Phase (radians)')
    ax2.set_title('Phase Spectrum')
    ax2.grid(True)
    plt.tight_layout()

# Classes
class Signals:
    """
    Class Signals
    This class provides functionality for managing several microwave signals.

    Attributes:
    -----------
    fmax        : float
        maximum frequency of signals
    bin_width   : float
        width of the spectral bins
    n_windows   : int
        number of signal windows (default=32)
        <n_windows> signals with same signal of interest but with different noises
    imped_ohms  : float
        Impedance, default=50
    temp_kelvin : float
        Temperature in Kelvin, default=temperature_default
    """
    @staticmethod
    def generate_signal_dbm(time, freq, power_dbm, phase, imped_ohms=50):
        """Generate a monotone signal."""
        amp_rms = dbm_to_voltage(power_dbm, imped_ohms)
        amp_pk = np.sqrt(2)*amp_rms
        s_ = amp_pk * np.sin(2*pi*freq*time + phase)
        return s_

    @staticmethod
    def generate_noise_dbm(shape, power_dbm, imped_ohms=50):
        """Generate noise, shape may be a tuple or a number."""
        amp = dbm_to_voltage(power_dbm, imped_ohms)
        return np.random.normal(0, amp, shape)

    def __init__(self, fmax, bin_width, n_windows=32, imped_ohms=50, temp_kelvin=temperature_default):
        """Initialize the Signals object."""
        self.fmax = fmax
        self.bin_width = bin_width
        self.imped_ohms = imped_ohms
        self.temp_kelvin = temp_kelvin
        self.n_windows = n_windows

        self.duration = 1/bin_width
        self.n_points = int(np.ceil(2.2 * fmax / bin_width))
        self.sampling_rate = self.n_points / self.duration
        self.freqs = np.fft.fftfreq(self.n_points, 1/self.sampling_rate)

        self.bw_hz = self.sampling_rate/2

        self.shape = (self.n_windows, self.n_points)
        self.size = np.prod(self.n_windows * self.n_points)

        self.time = np.linspace(0, self.duration, self.n_points, endpoint=False)
        self.sig2d = Signals.generate_noise_dbm(self.shape, -1000)  # very very low noise in order to avoid log errors

        self.spectrum_uptodate = False

    def __getitem__(self, key):
        if key in ('freqs', 'spectrums', 'spects_power', 'spects_phase'):
            self.compute_spectrum()
            return getattr(self, '_'+key)

    def compute_spectrum(self, force=False):
        """Compute the frequency spectrum of the signal."""
        if force or not self.spectrum_uptodate:
            self._freqs, self._spectrums = compute_spectrums(self.sig2d, self.sampling_rate)
            self._spects_power, self._spects_phase = get_spectrums_power_n_phase(self._freqs, self._spectrums, self.imped_ohms)
            self.spectrum_uptodate = True

    def get_arg_freq(self, freq):
        """Get the index of the given frequency."""
        return np.argmin(np.abs(self['freqs'] - freq))

    def add_signal(self, sigxd):
        """Add a signal to the current signal."""
        if sigxd.shape == self.sig2d.shape:
            self.sig2d += sigxd
        elif len(sigxd.shape) == 2 and sigxd.shape[0] == 1 and sigxd.shape[1] == self.sig2d.shape[1]:
            self.sig2d[:] += sigxd[0]
        elif len(sigxd.shape) == 1 and sigxd.shape[0] == self.sig2d.shape[1]:
            self.sig2d[:] += sigxd
        else:
            raise ValueError("Invalid input, sigxd.shape: %s" % (sigxd.shape))
        self.spectrum_uptodate = False

    def add_tone(self, freq, power_dbm, phase):
        """Add a tone to the signal."""
        self.add_signal(Signals.generate_signal_dbm(self.time, freq, power_dbm, phase, self.imped_ohms))

    def add_noise(self, power_dbm):
        """Add noise to the signal."""
        self.add_signal(Signals.generate_noise_dbm(self.shape, power_dbm))

    def add_thermal_noise(self, temp_kelvin=None):
        """Add thermal noise to the signal."""
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin