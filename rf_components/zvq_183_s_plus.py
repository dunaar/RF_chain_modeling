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
# python -m RF_chain_modeling.rf_components.zvq_183_s_plus
# ====================================================================================================

from RF_chain_modeling.rf_utils.csv_data_table import CSVDataTable
from RF_chain_modeling.rf_utils.rf_modeling import RF_Modelised_Component


csv_data = CSVDataTable('RF_chain_modeling/rf_components/zvq_183_s_plus.tsv', delim_field='\t', multi_df=False)

if __name__ == '__main__':
    print(csv_data)

gains_db  = csv_data.data['gain']
iip3s_dbm = csv_data.data['oip3'] - gains_db

cpnt = RF_Modelised_Component(
                               freqs      = csv_data.data['freq'], 
                               gains_db   = gains_db,
                               nfs_db     = csv_data.data['nf'],
                               phases_rad = csv_data.data['phase'],
                               iip3s_dbm  = iip3s_dbm,
                              )

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from RF_chain_modeling.rf_utils.rf_modeling import plot_signal_spectrum
    
    ####
    freqs, gains, phases, noise_figures = cpnt.assess_gain( fmin=max(0, csv_data.data['freq'][0] - 2e9), 
                                                            fmax=csv_data.data['freq'][-1] + 2e9)
    print(freqs)
    print(gains)
    plot_signal_spectrum(freqs, gains, phases, ylabel_power="Gain (dB)", ylabel_phase="Phase (radians)")
    plot_signal_spectrum(freqs, noise_figures, ylabel_power="NF (dB)")
    
    ####
    cpnt.assess_iipx()

    
    ####
    plt.show()