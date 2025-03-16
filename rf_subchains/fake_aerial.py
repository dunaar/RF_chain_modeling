#!/usr/bin/env python
# coding: utf-8

"""
Project: RF_chain_modeling
GitHub: https://github.com/dunaar/RF_chain_modeling
Auteur: Pessel Arnaud
"""

# ====================================================================================================
# Author: Pessel Arnaud
# Date: 2025-03-15
#
# python -m RF_chain_modeling.rf_subchains.fake_aerial
# ====================================================================================================

from  RF_chain_modeling.rf_utils.rf_modeling import RF_chain, Antenna_Component, Simple_Amplifier, HighPassFilter
import RF_chain_modeling.rf_components.zvq_183_s_plus as zvq_183_s_plus

ant  = Antenna_Component(freqs=(1e9, 20e9), gains_db=(1., 1.))
sp4t = Simple_Amplifier(gain_db=-1, nf_db=1, op1db_dbm=500, oip3_dbm=39)
hpf  = HighPassFilter(cutoff_freq=6e9, order=11, q_factor=0.7)
#amp = Simple_Amplifier(gain_db=26, nf_db=3, op1db_dbm=24, oip3_dbm=34)
amp  = zvq_183_s_plus.cpnt

cpnt = RF_chain( (ant, sp4t, hpf, amp) ) # cpnt is the generic name (understand chain as a macro-component)
#cpnt = RF_chain( (ant, ) ) # cpnt is the generic name (understand chain as a macro-component)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from RF_chain_modeling.rf_utils.rf_modeling import plot_signal_spectrum
    
    ####
    #freqs, gains, phases, noise_figures = cpnt.assess_gain(fmin=0.4e9, fmax=21e9)

    #plot_signal_spectrum(freqs, gains, phases, ylabel_power="Gain (dB)", ylabel_phase="Phase (radians)")
    #plot_signal_spectrum(freqs, noise_figures, ylabel_power="NF (dB)")
    
    ####
    cpnt.assess_iipx()

    
    ####
    plt.show()