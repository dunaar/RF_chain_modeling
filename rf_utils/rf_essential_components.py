#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Project: RF_chain_modeling
RF Signal Simulation and Analysis Framework

This module provides essential RF component modelisation like attenuators, amplifiers, cables, filters, and antennas.

Key Features:
- RF component modeling (attenuators, amplifiers, cables, filters, antennas).
- Signal processing with gain, noise figure, and non-linearities.

Author: Pessel Arnaud
Date: 2026-01-18
Version: 0.2
GitHub: https://github.com/dunaar/RF_chain_modeling
License: MIT
'''

__version__ = "0.2"

import logging
from typing import Optional, Tuple

import numpy as np
from numpy import pi, nan, isnan

from scipy.signal import butter, freqz

import matplotlib.pyplot as plt

from .rf_modeling import DEFAULT_TEMP_KELVIN, dbm_to_voltage, gain_db_to_gain, gain_to_gain_db, \
                         thermal_noise_power_dbm, nf_db_to_nf, infs_like, Signals, \
                         RF_Abstract_Base_Component, RF_Abstract_Modelised_Component, RF_Modelised_Component, RF_chain

logger = logging.getLogger(__name__)

# ====================================================================================================
# Attenuator Class
# ====================================================================================================
class Attenuator(RF_Abstract_Base_Component):
    '''Class representing an attenuator RF component.
    
    Attributes:
        temp_kelvin (float): Temperature in Kelvin.
        att_db (float): Attenuation in dB (positive value).
        nf_db (float): Noise figure in dB (equals attenuation for passive device).
        gain (float): Linear gain (less than 1 due to attenuation).
        nf (float): Linear noise figure contribution.
    '''

    def __init__(self, att_db: float, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        '''Initialize the attenuator with specified attenuation.
        
        Args:
            att_db (float): Attenuation in dB (positive value).
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
        '''
        self.temp_kelvin = temp_kelvin
        self.att_db = att_db
        self.nf_db = att_db  # Noise figure equals attenuation for a passive attenuator

        self.gain = gain_db_to_gain(-att_db)  # Convert attenuation to linear gain (< 1)
        self.nf = nf_db_to_nf(self.nf_db)

    def process_signals(self, signals: Signals, temp_kelvin: Optional[float] = None) -> None:
        '''Process signals by applying attenuation and adding thermal noise.
        
        Args:
            signals (Signals): Input signals object.
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to instance temperature.
        '''
        temp_kelvin = temp_kelvin if temp_kelvin is not None else self.temp_kelvin

        # Generate thermal noise based on temperature and bandwidth
        noise = Signals.generate_noise_dbm(signals.shape, thermal_noise_power_dbm(temp_kelvin, signals.bandwidth), signals.imped_ohms)
        signals.sig2d += self.nf * noise  # Add noise contribution
        signals.sig2d *= self.gain  # Apply attenuation

# ====================================================================================================
# Simple Amplifier Class
# ====================================================================================================
class Simple_Amplifier(RF_Abstract_Base_Component):
    '''Class representing a simple amplifier with gain, noise figure, and non-linearities.
    
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
    '''

    def __init__(self, gain_db: float, nf_db: float, op1db_dbm: float = 20, oip3_dbm: float = 10, iip2_dbm: float = 40, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        '''Initialize the amplifier with specified parameters.
        
        Args:
            gain_db (float): Gain in dB.
            nf_db (float): Noise figure in dB.
            op1db_dbm (float): Output 1dB compression point in dBm, defaults to 20 dBm.
            oip3_dbm (float): Output IP3 in dBm, defaults to 10 dBm.
            iip2_dbm (float): Input IP2 in dBm, defaults to 40 dBm.
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
        '''
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
        '''Process signals by applying gain, noise, and non-linear distortions.
        
        Args:
            signals (Signals): Input signals object.
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to instance temperature.
        '''
        temp_kelvin = temp_kelvin if temp_kelvin is not None else self.temp_kelvin

        # Add thermal noise
        noise = Signals.generate_noise_dbm(signals.shape, thermal_noise_power_dbm(temp_kelvin, signals.bandwidth), signals.imped_ohms)
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
# RF Cable Class
# ====================================================================================================
class RF_Cable(RF_Abstract_Modelised_Component):
    '''Class representing an RF cable with frequency-dependent insertion losses.
    
    Attributes:
        length_m (float): Cable length in meters.
        alpha (float): Attenuation coefficient (dB/sqrt(Hz)).
        insertion_losses_dB (float): Fixed insertion loss in dB.
    '''

    def __init__(self, length_m: float, alpha: float = 5.2e-06, insertion_losses_dB: float = 0, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        '''Initialize the RF cable with specified length and losses.
        
        Args:
            length_m (float): Cable length in meters.
            alpha (float): Attenuation coefficient in dB/sqrt(Hz), defaults to 5.2e-06.
            insertion_losses_dB (float): Fixed insertion loss in dB, defaults to 0.
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
        '''
        self.temp_kelvin = temp_kelvin
        self.length_m = length_m
        self.alpha = alpha
        self.insertion_losses_dB = insertion_losses_dB

    def get_rf_parameters_adapted_to_signals(self, signals: Signals, temp_kelvin: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''Get frequency-dependent gains and noise figures for the RF cable.
        
        Args:
            signals (Signals): Input signals object.
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to instance temperature.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Frequencies, gains, noise figures.
        '''
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin

        freqs    = signals.freqs[signals.freqs >= 0]
        gains_db = -self.insertion_losses_dB - self.alpha * np.sqrt(freqs) * self.length_m  # Frequency-dependent loss
        gains    = gain_db_to_gain(gains_db)
        nf__s    = nf_db_to_nf(-gains_db)  # Noise figure equals negative gain for passive device

        return freqs, gains, nf__s, infs_like(freqs), infs_like(freqs), infs_like(freqs)  # No op1db, iip3, iip2 for passive device

# ====================================================================================================
# RF Filter Classes
# ====================================================================================================
class RF_Filter(RF_Abstract_Modelised_Component):
    '''Class representing an RF filter with specified cutoff frequencies and order.
    
    Attributes:
        band_type (str): Band type ('high', 'low', or 'band').        
        cutoff_freq (float): Cutoff frequency in Hz.
        cutoff_freq_opt (float): Optional optimized cutoff frequency in Hz for band-pass filters.
        order (int): Filter order.
        q_factor (float): Quality factor (not used in Butterworth design).
        insertion_losses_dB (float): Insertion losses in dB.
        gain_losses (float): Linear gain factor due to insertion losses.
    '''
    HIGH_PASS = 'high'
    LOW_PASS  = 'low'
    BAND_PASS = 'band'

    def __init__(self, band_type: str, cutoff_freq: float, cutoff_freq_opt: float = nan, order: int = 1, q_factor: float = 1, insertion_losses_dB: float = 0, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        '''Initialize the RF filter with specified characteristics.
        
        Args:
            band_type (str): Band type (RF_Filter.HIGH_PASS 'high', RF_Filter.LOW_PASS 'low', or RF_Filter.BAND_PASS 'band').
            cutoff_freq (float)    : Cutoff frequency in Hz (for high-pass or low-pass filters, or 1st frequency for band-pass filter).
            cutoff_freq_opt (float): Optional optimized cutoff frequency in Hz for band-pass filters, this is the 2nd frequency. Defaults to None.
            order (int): Filter order, defaults to 1.
            q_factor (float): Quality factor, defaults to 1 (not used in Butterworth).
            insertion_losses_dB (float): Insertion losses in dB, defaults to 0.
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
        '''
        self.temp_kelvin = temp_kelvin
        self.band_type   = band_type

        if band_type == RF_Filter.BAND_PASS:
            # -- Ensure cutoff_freq_opt is provided for band-pass filters
            if isnan(cutoff_freq_opt):
                raise ValueError("For band-pass filters, cutoff_freq_opt must be provided.")
            
            # -- Ensure cutoff_freq is the lower frequency and cutoff_freq_opt is the higher frequency
            if cutoff_freq_opt > cutoff_freq:
                self.cutoff_freq     = cutoff_freq
                self.cutoff_freq_opt = cutoff_freq_opt
            else:
                self.cutoff_freq     = cutoff_freq_opt
                self.cutoff_freq_opt = cutoff_freq
            
            self.cutoff_freqs = (self.cutoff_freq, self.cutoff_freq_opt)
        else:
            self.cutoff_freq     = cutoff_freq
            self.cutoff_freq_opt = cutoff_freq_opt
            self.cutoff_freqs    = (self.cutoff_freq,)
        
        self.order    = order
        self.q_factor = q_factor

        self.insertion_losses_dB = insertion_losses_dB
        self.gain_losses         = gain_db_to_gain(-self.insertion_losses_dB)  # Convert insertion loss to linear gain

        # -- Cache initialization to optimize performance on repeated calls
        self._cache_ba_signature = None  # Tuple (order, cutoff_freq, sampling_rate)
        self._cache_fr_signature = None  # Tuple (start, stop, length)
        self._cache_ba           = None  # Coefficients b, a
        self._cache_params       = None  # gains, gains_db, nf__s

    def get_rf_parameters_adapted_to_signals(self, signals: Signals, temp_kelvin: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''Get frequency-dependent gains and noise figures for the RF filter.
        
        Args:
            signals (Signals): Input signals object.
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to instance temperature.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Frequencies, gains, noise figures, op1db, iip3, iip2.
        '''
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin

        freqs = signals.freqs[signals.freqs >= 0]

        # -- Design Butterworth RF filter
        #    Manage cache for filter coefficients (b, a) that depend solely on order, cutoff_freq, and sampling_rate
        sampling_rate, sampling_factor = signals.sampling_rate, 1
        while sampling_rate < self.cutoff_freqs[-1]*2: # Nyquist criterion, self.cutoff_freqs[-1] is the highest cutoff frequency
            sampling_factor *= 2
            sampling_rate   *= 2
        
        new_ba_signature = (self.order, self.cutoff_freq, sampling_rate)

        if new_ba_signature != self._cache_ba_signature:
            b, a = butter(self.order, self.cutoff_freqs, btype=self.band_type, fs=sampling_rate, output='ba')

            self._cache_ba_signature = new_ba_signature
            self._cache_ba           = (b, a)
            self._cache_params       = None  # Invalidate frequency response cache if coefficients change
        else:
            b, a = self._cache_ba

        new_fr_signature = (float(freqs[0]), float(freqs[-1]), len(freqs))

        if self._cache_params is None or new_fr_signature != self._cache_fr_signature:
            logger.debug(f'<{self.__class__.__name__}> signals.sampling_rate={signals.sampling_rate/1e9:.3f} GHz, sampling_rate={sampling_rate/1e9:.3f} GHz')
            logger.debug(f'<{self.__class__.__name__}> [{freqs[0]/1e9:.3f} GHz-{freqs[-1]/1e9:.3f} GHz], fc={self.cutoff_freq/1e9:.3f} GHz, sampling_rate={sampling_rate/1e9:.3f} GHz, order={self.order}, insertion_losses_dB={self.insertion_losses_dB} dB')
            w, h     = freqz(b, a, worN=freqs, fs=sampling_rate)
            gains    = h * self.gain_losses                  # Apply insertion loss
            gains_db = gain_to_gain_db(gains)
            nf__s    = nf_db_to_nf( np.clip(-gains_db, a_min=0., a_max=None) ) # Noise figure based on loss, Cap at 0 dB (passive device)

            self._cache_fr_signature = new_fr_signature
            self._cache_params = (gains, gains_db, nf__s)
        else:
            gains, gains_db, nf__s = self._cache_params

        return freqs, gains, nf__s, infs_like(freqs), infs_like(freqs), infs_like(freqs)  # No op1db, iip3, iip2 for passive device

class HighPass_Filter(RF_Filter):
    '''Class representing a high-pass filter with specified cutoff frequency and order.
    
    Attributes:
        cutoff_freq (float): Cutoff frequency in Hz.
        order (int): Filter order.
        q_factor (float): Quality factor (not used in Butterworth design).
        insertion_losses_dB (float): Insertion losses in dB.
        gain_losses (float): Linear gain factor due to insertion losses.
    '''

    def __init__(self, cutoff_freq: float, order: int = 1, q_factor: float = 1, insertion_losses_dB: float = 0, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        '''Initialize the high-pass filter with specified characteristics.
        
        Args:
            cutoff_freq (float): Cutoff frequency in Hz.
            order (int): Filter order, defaults to 1.
            q_factor (float): Quality factor, defaults to 1 (not used in Butterworth).
            insertion_losses_dB (float): Insertion losses in dB, defaults to 0.
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
        '''
        super().__init__(band_type=RF_Filter.HIGH_PASS, cutoff_freq=cutoff_freq, order=order, q_factor=q_factor, \
                         insertion_losses_dB=insertion_losses_dB, temp_kelvin=temp_kelvin)

class LowPass_Filter(RF_Filter):
    '''Class representing a low-pass filter with specified cutoff frequency and order.
    
    Attributes:
        cutoff_freq (float): Cutoff frequency in Hz.
        order (int): Filter order.
        q_factor (float): Quality factor (not used in Butterworth design).
        insertion_losses_dB (float): Insertion losses in dB.
        gain_losses (float): Linear gain factor due to insertion losses.
    '''

    def __init__(self, cutoff_freq: float, order: int = 1, q_factor: float = 1, insertion_losses_dB: float = 0, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        '''Initialize the low-pass filter with specified characteristics.
        
        Args:
            cutoff_freq (float): Cutoff frequency in Hz.
            order (int): Filter order, defaults to 1.
            q_factor (float): Quality factor, defaults to 1 (not used in Butterworth).
            insertion_losses_dB (float): Insertion losses in dB, defaults to 0.
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
        '''
        super().__init__(band_type=RF_Filter.LOW_PASS, cutoff_freq=cutoff_freq, order=order, q_factor=q_factor, \
                         insertion_losses_dB=insertion_losses_dB, temp_kelvin=temp_kelvin)

class BandPass_Filter(RF_Filter):
    '''
    Class representing a band-pass filter with specified center frequency and bandwidth.
    
    The filter is implemented as a Butterworth design, which provides maximally flat
    passband response. Only signals within the passband are transmitted; frequencies
    outside the band are attenuated.
    
    Attributes:
        cutoff_freq1 (float): Lower cutoff frequency in Hz.
        cutoff_freq2 (float): Upper cutoff frequency in Hz.
        order (int): Filter order (higher order = steeper roll-off).
        q_factor (float): Quality factor (not used in Butterworth design).
        insertion_losses_dB (float): Insertion losses in passband in dB.
        gain_losses (float): Linear gain factor due to insertion losses.
        cutoff_low (float): Lower cutoff frequency in Hz.
        cutoff_high (float): Upper cutoff frequency in Hz.
    '''
    
    def __init__(self, cutoff_freq1: float, cutoff_freq2: float, order: int = 2, 
                 q_factor: float = 1, insertion_losses_dB: float = 0, 
                 temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        '''
        Initialize the band-pass filter with specified characteristics.
        
        Args:
            cutoff_freq1 (float): Lower cutoff frequency in Hz of the passband.
            cutoff_freq2 (float): Upper cutoff frequency in Hz of the passband.
            order (int): Filter order, defaults to 2. Higher orders provide steeper roll-off.
            q_factor (float): Quality factor, defaults to 1 (not used in Butterworth design).
            insertion_losses_dB (float): Insertion losses in dB, defaults to 0.
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
        '''
        super().__init__(band_type=RF_Filter.BAND_PASS, cutoff_freq=cutoff_freq1, cutoff_freq_opt=cutoff_freq2, order=order, q_factor=q_factor, \
                         insertion_losses_dB=insertion_losses_dB, temp_kelvin=temp_kelvin)

# ====================================================================================================
# RF slope equalizer Class
# ====================================================================================================
class RF_slope_equalizer(RF_Abstract_Modelised_Component):
    def __init__(self, freq1: float, gain_db_at_freq1: float, freq2: float, gain_db_at_freq2: float, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        self.temp_kelvin             = temp_kelvin

        # -- Not used, just to keep attributes info
        self.freq1            = freq1
        self.gain_db_at_freq1 = gain_db_at_freq1
        
        self.freq2            = freq2
        self.gain_db_at_freq2 = gain_db_at_freq2

        # -- gains_db(f) = ax + b
        self.a  =  (gain_db_at_freq2 - gain_db_at_freq1)/(freq2 - freq1)
        self.b  = -(gain_db_at_freq2*freq1 - gain_db_at_freq1*freq2)/(freq2 - freq1)

    def get_rf_parameters_adapted_to_signals(self, signals, temp_kelvin=None):
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin

        freqs    = signals.freqs[signals.freqs >= 0]
        gains_db = self.a * freqs + self.b  # Frequency-dependent loss
        gains    = gain_db_to_gain(gains_db)
        nf__s    = nf_db_to_nf(-gains_db)  # Noise figure equals negative gain for passive device

        return freqs, gains, nf__s, infs_like(freqs), infs_like(freqs), infs_like(freqs)  # No op1db, iip3, iip2 for passive device


# ====================================================================================================
# Antenna Component Class
# ====================================================================================================
class Antenna_Component(RF_Modelised_Component):
    '''Class representing an antenna with frequency-dependent gain and phase.
       !!! With an antenna, the thermal noise of the incident signal received in the front of the antenna (dBmi)
           comes from the transmitter before free-space propagation.
           Considering a very strong attenuation of the signal due to free space propagation,
           we can assume that the incident signal is noise-free.
           However, if the signal has to include thermal noise, consider that the thermal noise is emitted by the transmitter,
           and must be attenuated by free space propagation.

    
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
    '''

    def __init__(self, freqs: np.ndarray, gains_db: np.ndarray, phases_rad: Optional[np.ndarray] = None, temp_kelvin: float = DEFAULT_TEMP_KELVIN) -> None:
        '''Initialize the antenna with specified gain and phase characteristics.
        
        Args:
            freqs (np.ndarray): Frequency points in Hz.
            gains_db (np.ndarray): Gains in dB at each frequency.
            phases_rad (Optional[np.ndarray]): Phases in radians, defaults to None.
            temp_kelvin (float): Temperature in Kelvin, defaults to 298.15 K.
        '''
        super().__init__(freqs, gains_db, phases_rad, temp_kelvin=temp_kelvin)

    def process_signals(self, signals: Signals, temp_kelvin: Optional[float] = None) -> None:
        '''Process signals by applying antenna gains and adding thermal noise.
        
        Args:
            signals (Signals): Input signals object.
            temp_kelvin (Optional[float]): Temperature in Kelvin, defaults to instance temperature.
        '''
        temp_kelvin = temp_kelvin if temp_kelvin is not None else self.temp_kelvin
        super().process_signals(signals, temp_kelvin)

        signals.sig2d += Signals.generate_noise_dbm(signals.shape, thermal_noise_power_dbm(temp_kelvin, signals.bandwidth))  # Add thermal noise

# ====================================================================================================
# Main Execution
# ====================================================================================================
def main() -> None:
    '''Main function to demonstrate the usage of the RF modeling classes.'''
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(module)s-%(funcName)s: %(message)s')

    # Example usage of the classes
    # Create a signal with noise and tones
    signal = Signals(20e9, 100e6)  # fmax = 40 GHz, bin_width = 10 MHz (duration = 1/bin_width = 100 ns)
    signal.add_noise(thermal_noise_power_dbm(signal.temp_kelvin, signal.bandwidth))  # Add thermal noise
    signal.add_tone(3e9, 0, 0)          # Add tone at 3 GHz, 0 dBm
    signal.add_tone(11e9, -55, pi / 4)  # Add tone at 11 GHz, -55 dBm
    signal.add_tone(17e9, -50, -pi / 3) # Add tone at 17 GHz, -50 dBm

    # Print RMS value
    logger.info( f"Initial RMS value: {signal.rms_dbm()} dBm" )

    # Plot temporal signal
    signal.plot_temporal(tmax=10e-9)

    # Plot spectrum
    signal.plot_spectrum()

    # Create an amplifier with non-linearities
    amplifier = Simple_Amplifier(gain_db=20, iip2_dbm=30, oip3_dbm=20, nf_db=5)

    # Process the signal through the amplifier
    amplifier.process(signal)

    # Print RMS value after amplification
    logger.info( f"RMS value after amplification: {signal.rms_dbm()} dBm" )

    # Plot temporal signal after amplification
    signal.plot_temporal(tmax=10e-9)

    # Plot spectrum after amplification
    signal.plot_spectrum()

    # Create a high-pass filter
    high_pass_filter = HighPass_Filter(cutoff_freq=6e9, order=5, q_factor=0.7)

    # Apply the high-pass filter to the amplified signal
    high_pass_filter.process(signal)

    # Print RMS value after filtering
    logger.info(f"RMS value after filtering: {signal.rms_dbm()} dBm")

    # Plot temporal signal after filtering
    signal.plot_temporal(tmax=10e-9)

    # Plot spectrum after filtering
    signal.plot_spectrum()

    # Testing components
    components = [
        Antenna_Component(freqs=(1e9, 20e9), gains_db=[3,7]),
        HighPass_Filter(cutoff_freq=6e9, order=5, q_factor=0.7),
        Simple_Amplifier(gain_db=15, iip2_dbm=30, oip3_dbm=20, nf_db=5),
        Attenuator(att_db=5),
        Simple_Amplifier(gain_db=15, iip2_dbm=30, oip3_dbm=20, nf_db=5),
        RF_Cable(length_m=10, insertion_losses_dB=0.5),
    ]

    chain = RF_chain(components)

    #for component in components+[chain]:
    for component in [components[1], chain]:
        logger.info(  "====================================================" )
        logger.info( f"=========== Assessing {component.__class__.__name__}" )
        logger.info(  "====================================================" )
        
        freqs, gains, phases, nf = component.assess_gain()
        plt.figure()
        plt.plot(freqs / 1e9, gains)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Gain (dB)')
        plt.title(f'Gain vs Frequency for {component.__class__.__name__}')
        
        results = component.assess_ipx(fmin=1e9, fmax=10e9, fstp=1e9)
        for freq, gain_db, op1db_dbm, iip3_dbm, oip3_dbm, iip2_dbm, oip2_dbm in results.T:
            logger.info( f"Frequency: {freq / 1e9} GHz, Gain: {gain_db} dB, OP1dB: {op1db_dbm} dBm, IIP2: {iip2_dbm} dBm, OIP3: {oip3_dbm} dBm" )
    plt.show()  # Display all plots

if __name__ == '__main__':
    main()
# ====================================================================================================
