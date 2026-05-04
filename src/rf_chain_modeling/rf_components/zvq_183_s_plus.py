#!/usr/bin/env python
# coding: utf-8

"""Module providing a specific commercial RF component model (ZVQ-183-S+).

Project: rf_chain_modeling
GitHub: https://github.com/dunaar/rf_chain_modeling
Author: Pessel Arnaud
Date: 2025-03-15
"""

import logging
from importlib import resources

import matplotlib.pyplot as plt

from rf_chain_modeling.rf_utils.csv_data_table import CSVDataTable
from rf_chain_modeling.rf_utils.rf_modeling import RF_Modelised_Component

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Open the static data file securely from the installed package
resource = resources.files("rf_chain_modeling.rf_components.data").joinpath("zvq_183_s_plus.tsv")

with resources.as_file(resource) as filepath:
    csv_data = CSVDataTable(filepath, delim_field="\t", multi_df=False)

gains_db  = csv_data.data['gain']
iip3s_dbm = csv_data.data['oip3'] - gains_db

# Component instantiation
cpnt = RF_Modelised_Component(
    freqs      = csv_data.data['freq'],
    gains_db   = gains_db,
    nfs_db     = csv_data.data['nf'],
    phases_rad = csv_data.data['phase'],
    op1ds_dbm  = csv_data.data['op1db'],
    iip3s_dbm  = iip3s_dbm,
    )

def main() -> None:
    """Main function to demonstrate the ZVQ-183-S+ component characteristics.

    This function processes the loaded CSV data and plots the component's 
    gain, phase, noise figure, and intermodulation intercept points.
    """
    from rf_chain_modeling.rf_utils.rf_modeling import plot_signal_spectrum

    logger.info("%s", csv_data)

    freqs, gains, phases, noise_figures = cpnt.assess_gain(
        fmin = max(0, csv_data.data['freq'][0] - 2e9),
        fmax = csv_data.data['freq'][-1] + 2e9
    )

    logger.info("freqs=%s", freqs)
    logger.info("gains=%s", gains)

    plot_signal_spectrum(
        freqs, gains, phases, 
        ylabel_power='Gain (dB)', 
        ylabel_phase='Phase (radians)'
        )

    plot_signal_spectrum(
        freqs, noise_figures, 
        ylabel_power='NF (dB)'
    )

    cpnt.assess_ipx(
        fmin = csv_data.data['freq'][0], 
        fmax = csv_data.data['freq'][-1]
    )

    plt.show()

if __name__ == '__main__':
    main()
