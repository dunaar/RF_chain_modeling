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
Date: 2025-03-15
Version: 0.1
GitHub: https://github.com/dunaar/RF_chain_modeling
License: MIT
'''

from math import pi
import logging
import matplotlib.pyplot as plt

from ..rf_utils.rf_modeling import Signals, RF_chain, thermal_noise_power_dbm
from ..rf_utils.rf_essential_components import Antenna_Component, HighPass_Filter, BandPass_Filter, Simple_Amplifier, Attenuator, RF_Cable

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------------
antenna       = Antenna_Component(freqs=(1e9, 20e9), gains_db=(3., 7.))

hp_filter     = HighPass_Filter(cutoff_freq=6e9, order=5, q_factor=0.7)

amplifier_lna = Simple_Amplifier(gain_db=16, oip3_dbm=30., nf_db=3)

attenuator    = Attenuator(att_db=5)

amplifier     = Simple_Amplifier(gain_db=20, iip2_dbm=30, oip3_dbm=40, nf_db=7)

rf_cable      = RF_Cable(length_m=10, insertion_losses_dB=0.5)

bp_filter     = BandPass_Filter(cutoff_freq1=9e9, cutoff_freq2=12e9, order=5, q_factor=0.7)
# -----------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------
components = [ antenna, hp_filter, amplifier_lna, attenuator, amplifier, rf_cable, bp_filter ]
chain      = RF_chain(components)
# -----------------------------------------------------------------------------------------------------


# ====================================================================================================
# Main Execution
# ====================================================================================================
def main() -> None:
    '''Main function to demonstrate the usage of the RF modeling classes.'''
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(module)s-%(funcName)s: %(message)s')

    # ---------------------------------------------------------------
    for component in [chain]:
    #for component in components+[chain]:
        logger.info(  "====================================================" )
        logger.info( f"=========== Assessing {component.__class__.__name__}" )
        logger.info(  "====================================================" )
        freqs, gains, phases, nf = component.assess_gain(fmin=1e9, fmax=19e9, fstp=1e9)

        for idx in range(len(freqs)):
            logger.info( f"Frequency: {freqs[idx] / 1e9} GHz, Gain: {gains[idx]} dB, Phase: {phases[idx]} rad, NF: {nf[idx]} dB" )

        results = component.assess_ipx(fmin=1e9, fmax=19e9, fstp=1e9)
        for freq, gain_db, op1db_dbm, iip3_dbm, oip3_dbm, iip2_dbm, oip2_dbm in results.T:
            logger.info( f"Frequency: {freq / 1e9} GHz, Gain: {gain_db} dB, OP1dB: {op1db_dbm} dBm, IIP2: {iip2_dbm} dBm, OIP3: {oip3_dbm} dBm" )
    # ---------------------------------------------------------------
        
    # ---------------------------------------------------------------
    # Example usage of the classes
    # Create a signal with noise and tones
    signal = Signals(20e9, 100e6)  # fmax = 40 GHz, bin_width = 100 MHz (duration of the time domain = 1/bin_width = 10 ns)

    signal.add_tone(3e9, 0, 0)          # Add tone at 3 GHz, 0 dBm
    signal.add_tone(11e9, -55, pi / 4)  # Add tone at 11 GHz, -55 dBm
    signal.add_tone(17e9, -50, -pi / 3) # Add tone at 17 GHz, -50 dBm

    # /!\ As the first compnent in the chain is antenna, no thermal noise is added before it
    #     With an antenna, the thermal noise is added by the component itself after signal processing it.
    #     -> See Antenna_Component class documentation for more details (RF_chain_modeling.rf_utils.rf_essential_components.Antenna_Component).
    # -> Uncomment the following line if the first component is not an antenna, for instance at antenna port before LNA
    # signal.add_noise(thermal_noise_power_dbm(signal.temp_kelvin, signal.bw_hz))  # Add thermal noise

    # Print RMS value
    logger.info( f"Initial RMS value: {signal.rms_dbm()} dBm" )

    # Plot temporal signal
    signal.plot_temporal(tmax=10e-9, title="Initial Temporal Signal")

    # Plot spectrum
    signal.plot_spectrum(title_power="Initial Power Spectrum", title_phase="Initial Phase Spectrum")

    # Process the signal through the RF chain
    chain.process_signals(signal)

    # Print RMS value after chain processing
    logger.info( f"RMS value after chain: {signal.rms_dbm()} dBm" )

    # Plot temporal signal after chain processing
    signal.plot_temporal(tmax=10e-9, title="Output Temporal Signal")

    # Plot spectrum after chain processing
    signal.plot_spectrum(title_power="Output Power Spectrum", title_phase="Output Phase Spectrum")

    plt.show()  # Display all plots

if __name__ == '__main__':
    main()
# ====================================================================================================
