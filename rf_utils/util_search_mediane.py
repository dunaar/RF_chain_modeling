#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Pessel Arnaud
Date: 2025-03-15
Version: 0.1
License: MIT
"""

__version__ = "0.1"

import logging
import numpy as np

logger = logging.getLogger(__name__)

# ====================================================================================================
# Functions
# ====================================================================================================
def search_mediane_for_slope(x, y, slope):
    """
    Find the median of the points where the slope is close to a given value.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    dy = np.diff(y) / np.diff(x)

    args = np.flatnonzero((dy > np.abs(slope)*0.8) & (dy < np.abs(slope)**1.2))
    args = np.unique(np.concatenate((args, args+1)))
    xargs, yargs = x[args], y[args]

    b_med = np.nan
    for _ in range(2):
        if len(xargs) > 0:
            args_to_keep = np.ones(len(xargs), dtype=bool)
            qn20, qp20   = 0., 0.

            bmin = (yargs - slope * xargs).min()
            bmax = (yargs - slope * xargs).max()
            stp  = min(0.1, (bmax-bmin)/100.)
            #logger.info('bmin, bmax, stp:', bmin, bmax, stp)

            for _b in np.arange(bmin, bmax+0.1*stp, stp):
                residus = yargs - (slope * xargs + _b)

                qn = np.count_nonzero(residus < 0) / len(residus)
                qp = np.count_nonzero(residus > 0) / len(residus)
                #logger.info('_b, qn, qp:', _b, qn*100., qp*100.)
                
                if qn > qn20 and qn < 0.2:
                    qn20 = qn
                    args_to_keep &= (residus >= 0)
                elif qp > qp20 and qp < 0.2:
                    qp20 = qp
                    args_to_keep &= (residus <= 0)

            xargs = xargs[args_to_keep]
            yargs = yargs[args_to_keep]

            b_med = np.polyfit(xargs, yargs - slope * xargs, 0)[0] if len(xargs) > 0 else np.nan

            args_close = (np.abs(yargs - (slope * xargs + b_med)) < 2.)
            if np.count_nonzero(args_close) > 2:
                #logger.info( f'args_close: {args_close}' )
                xargs = xargs[args_close]
                yargs = yargs[args_close]

        logger.debug( f'len(x), len(xargs): {len(x), len(xargs)}' )
    b_med = np.polyfit(xargs, yargs - slope * xargs, 0)[0] if len(xargs) > 0 else np.nan

    '''
    import matplotlib.pyplot as plt    
    plt.figure()    
    plt.plot(x, y                , 'x' , label='Data points')
    plt.plot(xargs, yargs        , 'x' , label='Kept points')
    plt.plot(x, slope * x + b_med, 'r-', label='Fitted line')
    plt.grid()
    plt.legend()
    plt.show()  # Display all plots
    '''
    
    return slope, b_med


# ====================================================================================================
# Main Execution
# ====================================================================================================

def main(points_repartition=(5, 1, 5, 2, 3, 1, 2), slope = 2.0) -> None:
    """Main function to test."""
    import matplotlib.pyplot as plt    

    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(module)s-%(funcName)s: %(message)s')

    logger.info( f'points_repartition: {points_repartition}' )
    nb_points = sum(points_repartition)

    x = np.arange(nb_points) + np.random.rand(nb_points) * 0.1

    slope_var = 1
    y = np.zeros(0, dtype=float)
    for nb in points_repartition:
        logger.info( f'nb: {nb}, slope: {slope+slope_var}' )
        y = np.concatenate( (y, np.full(nb, slope+slope_var)) )
        slope_var = (slope_var+2)%3-1
    
    y = np.cumsum(y) + np.random.rand(nb_points) * 0.2

    slope, b_med = search_mediane_for_slope(x, y, slope)

    logger.info( f"Slope: {slope}, Median: {b_med}" )

    plt.figure()    
    plt.plot(x, y                , 'x' , label='Data points')
    plt.plot(x, slope * x + b_med, 'r-', label='Fitted line')
    plt.grid()
    plt.legend()
    plt.show()  # Display all plots

if __name__ == '__main__':
    main()
# ====================================================================================================
