#!/usr/bin/env python
# coding: utf-8

"""
Project: rf_chain_modeling
GitHub: https://github.com/dunaar/rf_chain_modeling
Auteur: Pessel Arnaud
"""

# ====================================================================================================
# Author: Pessel Arnaud
# Date: 2025-03-15
#
# python -m rf_chain_modeling.rf_components.zvq_183_s_plus
# ====================================================================================================

import logging
from pathlib import Path

from rf_chain_modeling.rf_utils.csv_data_table import CSVDataTable
from rf_chain_modeling.rf_utils.rf_modeling import RF_Modelised_Component

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
csv_data = CSVDataTable(_HERE / "zvq_183_s_plus.tsv", delim_field="\t", multi_df=False)

if __name__ == '__main__':
    logger.info("%s", csv_data)

gains_db  = csv_data.data['gain']
iip3s_dbm = csv_data.data['oip3'] - gains_db

cpnt = RF_Modelised_Component(
                               freqs      = csv_data.data['freq'], 
                               gains_db   = gains_db,
                               nfs_db     = csv_data.data['nf'],
                               phases_rad = csv_data.data['phase'],
                               op1ds_dbm  = csv_data.data['op1db'],
                               iip3s_dbm  = iip3s_dbm,
                              )

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from rf_chain_modeling.rf_utils.rf_modeling import plot_signal_spectrum

    ####
    freqs, gains, phases, noise_figures = cpnt.assess_gain( fmin=max(0, csv_data.data['freq'][0] - 2e9), 
                                                            fmax=csv_data.data['freq'][-1] + 2e9)
    logger.info("freqs=%s", freqs)
    logger.info("gains=%s", gains)
    plot_signal_spectrum(freqs, gains, phases, ylabel_power="Gain (dB)", ylabel_phase="Phase (radians)")
    plot_signal_spectrum(freqs, noise_figures, ylabel_power="NF (dB)")

    ####
    cpnt.assess_ipx(
        fmin=csv_data.data['freq'][ 0],
        fmax=csv_data.data['freq'][-1]
    )

    ####
    plt.show()
