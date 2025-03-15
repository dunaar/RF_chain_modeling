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
# python -m RF_chain_modeling.rf_components.zvq-183-s+
# ====================================================================================================

from RF_chain_modeling.rf_utils.csv_data_table import CSVDataTable
from RF_chain_modeling.rf_utils.rf_modeling import RF_modelised_component



csv_data = CSVDataTable('RF_chain_modeling/rf_components/zvq-183-s+.tsv', delim_field='\t', multi_df=False)

if __name__ == '__main__':
    print(csv_data)



cpnt = RF_modelised_component(
                               freqs      = csv_data.data['freq'], 
                               gains_db   = csv_data.data['gain'],
                               nfs_db     = csv_data.data['nf'],
                               phases_rad = csv_data.data['phase'] * CSVDataTable.D2R,
                               nominal_gain_for_im_db = float(csv_data.attrs['nominal_gain(dB)']),
                               op1db_dbm  = float(csv_data.attrs['nominal_gain(dB)']),
                               oip3_dbm   = float(csv_data.attrs['oip3(dbm)']),
                              )

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from RF_chain_modeling.rf_utils.rf_modeling import plot_signal_spectrum
    
    ####
    freqs, gains, phases, noise_figures = cpnt.assess_gain()
    plot_signal_spectrum(freqs, gains        , phases)
    plot_signal_spectrum(freqs, noise_figures, phases)
    
    ####
    cpnt.assess_iipx()

    
    ####
    plt.show()