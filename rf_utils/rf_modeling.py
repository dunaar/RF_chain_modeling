#!/usr/bin/env python
# coding: utf-8

# ====================================================================================================
import copy

import numpy as np
from   numpy import pi

from scipy.signal      import butter, sosfilt, freqz
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
# ====================================================================================================


# ====================================================================================================
k_B = 1.38e-23  # Constante de Boltzmann en Joules par Kelvin
temperature_default = 290
# ====================================================================================================

# ====================================================================================================
# ------------------------------------------------------------------------------------------
D2R, R2D = np.pi/180., 180./np.pi
units = {'hz' :(1., 'Hz' ), 'khz':(1e3, 'Hz' ), 'mhz':(1e6, 'Hz'), 'ghz':(1e9, 'Hz'),
         'rad':(1., 'rad'), 'deg':(D2R, 'rad'),
         'db' :(1., 'dB'),
        }
def convert_unit(unit):
    unit_factor = units.get(unit.lower(), None)
    if unit_factor is None: unit_factor = (1., unit)
    return unit_factor

def extrapolate_nan(freq, arr):
    argsnan = np.where(np.isnan(arr))[0]
    
    if len(argsnan) > 0:
        argsnotnan = np.where(np.isnan(arr)==False)[0]
        arr[argsnan] = np.interp(freq[argsnan], freq[argsnotnan], arr[argsnotnan])

def read_from_csv(filepath, delim_field='\t', multi_df=False, delim_decim='.', delim_cmt='#'):
    '''Columns must have titles each one'''
    comments  = []
    dict_data = { 'comments': comments, 'titles': None, 'units': [], 'data': None, 'length': 0 }

    lines = []
    with open(filepath, 'r') as of: lines = of.readlines()
    #print(f"Error: File '{filepath}' not found.")

    length = 0
    for idx_line, line in enumerate(lines):
        try:
            # -- Keeping comments #TOTO
            line = line.split(delim_cmt)
            if len(line) > 1: comments.append( delim_cmt.join(line[1:]) )
            #else            : comments.append( '' )
            line = line[0] # Keeping line except comments
    
            # -- Treatin the fields
            if line.strip() == '': continue
    
            if multi_df:
                line = delim_field.join(s_ for s_ in line.split(delim_field) if s_ != '') 
                
            fields = line.split(delim_field)
    
            if dict_data['titles'] is None:
                # -- No key in dict_data means fields are the column titles
                dict_data['titles'] = []
                
                for idx, field in enumerate(fields):
                    field = field.strip()
                    field = field.strip('"')
                    if field == '': break
                    
                    split = field.split('(')
                    title = split[0].strip().lower()
                    unit  = split[1].split(')')[0].strip() if len(split) > 1 else ''
    
                    suffix = ''
                    while (title+suffix) in dict_data:
                        suffix = '_%d'%(eval(suffix[1:])+1) if suffix != '' else '_2'
                    title = title+suffix

                    dict_data['titles'].append(title) 
                    dict_data[title] = {'unit': unit, 'data':[]}
                
                dict_data['titles'] = tuple(dict_data['titles'])
                continue
            
            for idx, title in enumerate(dict_data['titles']):
                value = None
                
                if idx < len(fields):
                    try:
                        value = float( fields[idx].replace(delim_decim, '.') )
                    except:
                        print('Error! File <%s>, line %03d, field %02d: bad conversion'%(filepath, idx_line+1, idx+1))
                        print('%s'%(lines[idx_line]), end='')
                        print('<%s>'%(fields[idx]))
                        value = float('nan')
                else:
                    print('Error! File <%s>, line %03d, field %02d: missing values'%(filepath, idx_line+1, idx+1))
                    print(lines[idx_line])
                    value = float('nan')
    
                if value is None:
                    value = float('nan')
                    print(lines[idx_line])
                    print('Error! File <%s>, line %03d, field %02d: something was wrong'%(filepath, idx_line+1, idx+1))
                
                dict_data[title]['data'].append(value)
            # End of for loop on titles
            length += 1
        except:
            print('%03d: <%s>'%(idx_line, line))
            raise

        dict_data['length'] = length
    # End of for loop on lines

    dt                 = np.dtype( [(title.lower(), 'float') for title in dict_data['titles']] )
    dict_data['data']  = np.zeros(length, dtype=dt)

    for title in dict_data['titles']:
        factor, unit             = convert_unit(dict_data[title]['unit'])
        
        dict_data[title]['unit'] = unit
        dict_data['units'].append(unit)

        dict_data['data'][title] = factor * np.array(dict_data[title]['data'])
        dict_data[title]['data'] = dict_data['data'][title]

    for title in dict_data['titles']:
        extrapolate_nan(dict_data['freq']['data'], dict_data[title]['data'])
    
    dict_data['units'] = tuple(dict_data['units'])
    
    return dict_data

def write_to_csv(filepath, dict_data, delim_field='\t', delim_decim='.', delim_cmt='#'):
    with open(filepath, 'w') as of:
        for cmt in dict_data['comments']:
            of.write('%s %s'%(delim_cmt, cmt))

        for idx_title, title in enumerate(dict_data['titles']):
            if idx_title > 0: of.write(delim_field)
            of.write('%s(%s)'%(title, dict_data['units'][idx_title]))
        of.write('\n')
        
        for idx_arr in range(dict_data['length']):
            for idx_title, title in enumerate(dict_data['titles']):
                if idx_title > 0: of.write(delim_field)
                of.write('%s'%(dict_data[title]['data'][idx_arr]))
            of.write('\n')
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
def dbm_to_watts(power_dbm):
    return 10 ** (power_dbm / 10) / 1000  # Convertir dBm en watts

def watts_to_voltage(power_watts, imped_ohms=50):
    return np.sqrt(power_watts * imped_ohms)

def dbm_to_voltage(power_dbm, imped_ohms=50):
    return watts_to_voltage(dbm_to_watts(power_dbm), imped_ohms)

def gain_db_to_gain(gain_db):
    return 10**(gain_db/20.)

def gain_to_gain_db(gain):
    return 20*np.log10(np.abs(gain))

def nf_db_to_nf(nf_db):
    return np.sqrt( 10**(nf_db/10) - 1)

def nf_to_nf_db(nf):
    return 10*np.log10(nf**2 + 1)

def mul_nfs(nf1, nf2):
    "Multiply noise factors"
    return np.sqrt( (nf1**2 + 1) * (nf2**2 + 1) - 1 )
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
def voltage_to_watts(voltage, imped_ohms=50):
    return voltage**2 / imped_ohms

def watts_to_dbm(power_watts):
    return 10 * np.log10(power_watts * 1000)

def voltage_to_dbm(voltage, imped_ohms=50):
    return watts_to_dbm(voltage_to_watts(voltage, imped_ohms))
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
def calculate_rms(signal):
    return np.sqrt(np.mean(np.abs(signal)**2))

def calculate_rms_dbm(signal):
    return voltage_to_dbm(calculate_rms(signal))
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
def thermal_noise_power_dbm(temp_kelvin, bw_hz):
    return watts_to_dbm(k_B * temp_kelvin * bw_hz)
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
def compute_spectrums(sigxd, sampling_rate):
    '''Calcul des spectres de fréquence'''
    if len(sigxd.shape) == 1:
        sig2d = sigxd.reshape(1, signals.shape[0])
    else:
        sig2d = sigxd
    
    n_windows = sig2d.shape[0]
    n_points  = sig2d.shape[1]
    
    freqs     = np.fft.fftfreq(n_points, 1/sampling_rate)
    spectrums = np.fft.fft(sig2d, axis=1) / n_points
    
    return freqs, spectrums

def get_spectrums_power_n_phase(freqs, spectrums, n_windows=1, imped_ohms=50):
    '''
    Spectre de fréquence en dBm et en phase
    Énergie vs Puissance :
    La DFT calcule les coefficients qui représentent l'énergie à chaque fréquence.
    Pour obtenir une puissance par canal comparable, il faut normaliser (diviser) par le nombre d'échantillons N la DFT.
    Sans normaliser, pour obtenir la puissance moyenne (RMS) du spectre, il faut calculer le RMS comme dans le domaine temporel :
    - racine de la  **moyenne** des carrés.
    Quand le spectre est normalisé (donc déjà divisé par N), la puissance moyenne RMS du spectre est:
    - racine de la **somme** des carrés.

    Interprétation des Composantes Fréquentielles :
    La DFT d'un signal réel produit un spectre symétrique. Les composantes négatives et positives doivent être correctement interprétées.
    Pour un signal réel, l'énergie est répartie entre les fréquences positives et négatives.
    '''
    spects_amp   = abs(spectrums)
    spects_power = voltage_to_dbm(spects_amp, imped_ohms)
    spects_phase = np.angle(spectrums)  # Calcul de la phase

    #idx_max = np.argmax(freqs)+1
    #idx_pmax = spect_power[:idx_max].argmax()
    #print('Fréquence et puissance (dBm) du signal max:', freqs[idx_pmax], spect_power[idx_pmax])
    #print('Puissance moyenne (dBm) du spectre:', voltage_to_dbm( np.sqrt(np.sum(spect_amp**2)) ))
    return spects_power, spects_phase
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
def plot_temporal_signal(time, sigxd, tmax=None, tmin=None):
    '''Visualisation temporelle'''
    time = np.array(time)
    
    tmin = tmin if tmin else 0
    tmax = tmax if tmax else time.max()
    
    idx_min = np.argmin( np.abs(time-tmin) )
    idx_max = np.argmin( np.abs(time-tmax) )

    if   (time[-1] - time[0]) > 10 :
        unit = 's'
    elif (time[-1] - time[0]) > 10e-3 :
        time = time * 1e3
        unit = 'ms'
    elif (time[-1] - time[0]) > 10e-6 :
        time = time * 1e6
        unit = 'us'
    else:
        time = time * 1e9
        unit = 'ns'
    
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.subplots(1, 1, sharex=True)

    if   len(sigxd.shape) == 1:
        ax1.plot(time[idx_min:idx_max], sigxd[idx_min:idx_max])
    elif len(sigxd.shape) == 2:
        for idx in range(sigxd.shape[0]):
            ax1.plot(time[idx_min:idx_max], sigxd[idx, idx_min:idx_max])
    
    ax1.set_xlabel('Time (%s)'%(unit))
    ax1.set_ylabel('Amplitude')
    #ax1.set_title('Signal')
    #ax1.legend()
    ax1.grid(True)
    plt.tight_layout()

def plot_signal_spectrum(freqs, spectrum_power, spectrum_phase):
    '''Visualisation du spectre de fréquence en dBm et de la phase'''
    freqs   = np.array(freqs)
    idx_max = np.argmax(freqs)+1

    unit = ''
    if   freqs.max()-freqs[0] > 10e9 :
        freqs = freqs / 1e9
        unit = 'GHz'
    elif freqs.max()-freqs[0] > 10e6 :
        freqs = np.array(freqs) / 1e6
        unit = 'MHz'
    elif freqs.max()-freqs[0] > 10e3 :
        freqs = np.array(freqs) / 1e3
        unit = 'kHz'
    else:
        unit = 'Hz'
    
    fig = plt.figure(figsize=(12, 6))
    ax1, ax2 = fig.subplots(2, 1, sharex=True)

    if   len(spectrum_power.shape) == 1:
        ax1.plot(freqs[:idx_max], spectrum_power[:idx_max])
        ax2.plot(freqs[:idx_max], spectrum_phase[:idx_max])
    elif len(spectrum_power.shape) == 2:
        for idx in range(spectrum_power.shape[0]):
            ax1.plot(freqs[:idx_max], spectrum_power[idx][:idx_max])
            ax2.plot(freqs[:idx_max], spectrum_phase[idx][:idx_max])
        
    ax1.set_ylim(spectrum_power.min()-1., spectrum_power.max()+1.)
    ax1.set_xlabel('Fréquence (%s)'%(unit))
    ax1.set_ylabel('Puissance (dBm)')
    ax1.set_title('Spectre de Fréquence en dBm')
    #ax1.legend()
    ax1.grid(True)
    #ax1.set_xscale('log')
    #plt.xlim(0, freqs[freqs.size//2])  # Limiter l'axe des fréquences pour une meilleure visualisation
    
    ax2.set_ylim(-pi, pi)
    ax2.set_xlabel('Fréquence (%sHz)'%(unit))
    ax2.set_ylabel('Phase (radians)')
    ax2.set_title('Spectre de Phase')
    #ax2.legend()
    ax2.grid(True)
    #plt.xlim(0, freqs[freqs.size//2])  # Limiter l'axe des fréquences pour une meilleure visualisation
    
    plt.tight_layout()
# ------------------------------------------------------------------------------------------
# ====================================================================================================


# ====================================================================================================
class Signals:
    '''
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
    '''
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def generate_signal_dbm(time, freq, power_dbm, phase, imped_ohms=50):
        '''Generate a monotone signal'''
        amp_rms = dbm_to_voltage(power_dbm, imped_ohms)
        amp_pk  = np.sqrt(2)*amp_rms
        s_  = amp_pk * np.sin(2*pi*freq*time + phase)
        return s_
    
    @staticmethod
    def generate_noise_dbm(shape, power_dbm, imped_ohms=50):
        '''Generate noise, shape may be a tuple or a number'''
        amp  = dbm_to_voltage(power_dbm, imped_ohms)
            
        return np.random.normal(0, amp, shape)
    # ------------------------------------------------------------------------------------------
    
    # ------------------------------------------------------------------------------------------
    # Initialization et Méthodes
    # ------------------------------------------------------------------------------------------
    def __init__(self, fmax, bin_width, n_windows=32, imped_ohms=50, temp_kelvin=temperature_default):
        self.fmax          = fmax
        self.bin_width     = bin_width
        self.imped_ohms    = imped_ohms
        self.temp_kelvin   = temp_kelvin
        self.n_windows     = n_windows
        
        self.duration      = 1/bin_width
        self.n_points      = int( np.ceil(2.2 * fmax / bin_width) )
        self.sampling_rate = self.n_points / self.duration
        self.freqs         = np.fft.fftfreq(self.n_points, 1/self.sampling_rate)
        
        self.bw_hz         = self.sampling_rate/2
        
        self.shape = (self.n_windows, self.n_points)
        self.size  = np.prod(self.n_windows *self.n_points)
        
        self.time  = np.linspace(0, self.duration, self.n_points, endpoint=False)
        self.sig2d = Signals.generate_noise_dbm(self.shape, -1000) # very very low noise in order to avoid log errors

        self.spectrum_uptodate = False

    def __getitem__(self, key):
        if key in ('freqs', 'spectrums', 'spects_power', 'spects_phase'):
            self.compute_spectrum()
            return getattr(self, '_'+key)

    def compute_spectrum(self, force=False):
        if force or not self.spectrum_uptodate:
            self._freqs       , self._spectrums    = compute_spectrums(self.sig2d, self.sampling_rate)
            self._spects_power, self._spects_phase = get_spectrums_power_n_phase(self._freqs, self._spectrums, self.imped_ohms)
            self.spectrum_uptodate = True
    
    def get_arg_freq(self, freq):
        return np.argmin( np.abs(self['freqs']-freq) )
    
    def add_signal(self, sigxd):
        if sigxd.shape == self.sig2d.shape:
            self.sig2d += sigxd
        elif len(sigxd.shape) == 2 and sigxd.shape[0] == 1 and sigxd.shape[1] == self.sig2d.shape[1]:
            self.sig2d[:] += sigxd[0]
        elif len(sigxd.shape) == 1 and sigxd.shape[0] == self.sig2d.shape[1]:
            self.sig2d[:] += sigxd
        else:
            raise ValueError("Invalid input, sigxd.shape: %s"%(sigxd.shape))
        
        self.spectrum_uptodate = False
    
    def add_tone(self, freq, power_dbm, phase):
        self.add_signal( Signals.generate_signal_dbm(self.time, freq, power_dbm, phase, self.imped_ohms) )

    def add_noise(self, power_dbm):
        self.add_signal( Signals.generate_noise_dbm(self.shape, power_dbm) )

    def add_thermal_noise(self, temp_kelvin=None):
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin 
        self.add_noise( thermal_noise_power_dbm(temp_kelvin, self.bw_hz) )

    def rms_dbm(self):
        return calculate_rms_dbm(self.sig2d)
        
    def plot_temporal(self, tmax=None, tmin=None):
        idx_min = int((tmin if tmin else 0            ) * self.sampling_rate)
        idx_max = int((tmax if tmax else self.duration) * self.sampling_rate)
        
        plot_temporal_signal(self.time[idx_min:idx_max], self.sig2d[:, idx_min:idx_max])

    def plot_spectrum(self, fmax=None, fmin=None):
        #idx_min = int((fmin if fmin else 0            ) * self.sampling_rate)
        #idx_max = int((fmax if fmax else self.duration) * self.sampling_rate)
        
        self.compute_spectrum()
        plot_signal_spectrum(self['freqs'], self['spects_power'], self['spects_phase'])
# ====================================================================================================


# ====================================================================================================
# ------------------------------------------------------------------------------------------
class RF_Component(object):
    def __init__(self):
        pass

    def process_signals(self, signals, temp_kelvin=None):
        ''' **INPLACE** Generic method to be replaced by subclasses'''
        return None
    
    def process(self, signals, temp_kelvin=None, inplace=True):
        if not inplace:
            signals = copy.deepcopy(signals)

        self.process_signals(signals, temp_kelvin=temp_kelvin)
        signals.spectrum_uptodate = False

        return signals if not inplace else None
    
    def assess_gain(self, fmin=400e6, fmax=19e9, step=100e6, temp_kelvin=temperature_default):
        '''Assess the gain and phase versus frequency of the RF component'''
        bin_width = step/2
        n_windows = 128
        
        # -- Initializing noisy signals
        noisy_signals = Signals(fmax, bin_width, n_windows=n_windows, imped_ohms=50, temp_kelvin=temp_kelvin)
        noisy_signals.add_thermal_noise(temp_kelvin=temp_kelvin)
        
        #processed_noisy_signal = self.process(noisy_signal, temp_kelvin=temp_kelvin, inplace=False)
        #processed_noisy_signals.append(processed_noisy_signal)
        #processed_noisy_signals = np.array( processed_noisy_signals )
        
        # -- Gains and Phases ---------------------------------------------------
        freqs_for_test = np.linspace(fmin, fmax, int((fmax-fmin)/step)+1)
        gains = np.zeros_like(freqs_for_test)
        phass = np.zeros_like(freqs_for_test)
        n_fgs = np.zeros_like(freqs_for_test)

        for idx_frq, freq in enumerate(freqs_for_test):
            # -- Computating Gains & Phases
            clear_signal = Signals(fmax, bin_width, n_windows=1, imped_ohms=50, temp_kelvin=temp_kelvin)
            clear_signal.add_tone(freq, -50, 0)
            
            arg_frq_pos = clear_signal.get_arg_freq( freq)
            arg_frq_neg = clear_signal.get_arg_freq(-freq)

            gains[idx_frq] = clear_signal['spects_power'][0, arg_frq_pos]
            phass[idx_frq] = clear_signal['spects_phase'][0, arg_frq_pos]
            
            proc_clear_signal = self.process(clear_signal, temp_kelvin=temp_kelvin, inplace=False)
            
            gains[idx_frq] = proc_clear_signal['spects_power'][0, arg_frq_pos] - gains[idx_frq]
            phass[idx_frq] = proc_clear_signal['spects_phase'][0, arg_frq_pos] - phass[idx_frq]

            # -- Computating Noise Figure
            signals      = copy.deepcopy(noisy_signals)
            signals.add_signal(clear_signal.sig2d)
            
            spects_befor = (        signals['spectrums'][:, arg_frq_neg] +
                            np.conj(signals['spectrums'][:, arg_frq_pos]) )/2
            
            self.process(signals, temp_kelvin=temp_kelvin)
            
            spects_after = (        signals['spectrums'][:, arg_frq_neg] +
                            np.conj(signals['spectrums'][:, arg_frq_pos]) )/2

            #print(voltage_to_dbm(np.abs(spects_befor)))
            #print(voltage_to_dbm(np.abs(spects_after)))
            pwr_sig_befor = voltage_to_dbm( np.abs( spects_befor.mean() ) )
            pwr_sig_after = voltage_to_dbm( np.abs( spects_after.mean() ) )
            #print('pwr_sig_befor, pwr_sig_after', pwr_sig_befor, pwr_sig_after)
            
            pwr_nse_befor = voltage_to_dbm( np.sqrt( spects_befor.var() ) )
            pwr_nse_after = voltage_to_dbm( np.sqrt( spects_after.var() ) )
            #print('pwr_nse_befor, pwr_nse_after', pwr_nse_befor, pwr_nse_after)
            
            n_fgs[idx_frq] = (pwr_sig_befor-pwr_nse_befor) - (pwr_sig_after-pwr_nse_after)

        phass[phass<-pi] += 2*pi
        phass[phass> pi] -= 2*pi
        # -----------------------------------------------------------------------
        
        return freqs_for_test, gains, phass, n_fgs
    
    def assess_iipx(self, fc=9e9, df=400e6, temp_kelvin=temperature_default):
        '''Assess the IIP2 and IIP3 (Input Intercept Point of order 2 and 3) of the RF component'''
        fmax      = 2.5 * fc
        bin_width = df / 32
        n_windows = 8
        
        f1        =   fc - df
        f2        =   fc + df
        f1pf2     =   f1 + f2
        df1mf2    = 2*f1 - f2
        df2mf1    = 2*f2 - f1

        input_power, outpt_power, im2___power, im3___power = [], [], [], []

        op1db_dbm, iip2_dbm, oip3_dbm = None, None, None

        power_dbm   = -50
        to_continue = True
        while to_continue:
            #print(power_dbm)
            
            # -- Initializing bitone signals
            bitone_signals = Signals(fmax, bin_width, n_windows=n_windows, imped_ohms=50, temp_kelvin=temp_kelvin)
            bitone_signals.add_tone(f1, power_dbm, 0)
            bitone_signals.add_tone(f2, power_dbm, 0)
    
            # -- Indexes of frequencies in spectrum ---------------------------------
            arg_frq_f1     = bitone_signals.get_arg_freq(f1    )
            arg_frq_f2     = bitone_signals.get_arg_freq(f2    )
            arg_frq_f1pf2  = bitone_signals.get_arg_freq(f1pf2 )
            arg_frq_df1mf2 = bitone_signals.get_arg_freq(df1mf2)
            arg_frq_df2mf1 = bitone_signals.get_arg_freq(df2mf1)
            
            input_power.append( np.mean( (bitone_signals['spects_power'][:, arg_frq_f1]+
                                          bitone_signals['spects_power'][:, arg_frq_f2])/2 ) )
            
            # -- Processing the insertion_losses_dB ----------------------------------------------
            self.process(bitone_signals, temp_kelvin=temp_kelvin)
            #bitone_signals.plot_spectrum()

            # -- Append values in tables --------------------------------------------
            outpt_power.append( np.mean( (bitone_signals['spects_power'][:, arg_frq_f1]+
                                          bitone_signals['spects_power'][:, arg_frq_f2])/2 ) )
            
            im2___power.append( np.mean(  bitone_signals['spects_power'][:, arg_frq_f1pf2] ) )
            
            im3___power.append( np.mean( (bitone_signals['spects_power'][:, arg_frq_df1mf2]+
                                          bitone_signals['spects_power'][:, arg_frq_df2mf1])/2 ) )
            # -----------------------------------------------------------------------

            if len(input_power) >= 2:
                delta_input_power = input_power[-1] - input_power[-2]
                delta_outpt_power = outpt_power[-1] - outpt_power[-2]
                delta_im2___power = im2___power[-1] - im2___power[-2]
                delta_im3___power = im3___power[-1] - im3___power[-2]

                #print(input_power, outpt_power, im2___power, im3___power)
                #print(100*np.abs(delta_im3___power/3-delta_input_power) / delta_input_power,
                #      100*np.abs(delta_im2___power/2-delta_input_power) / delta_input_power,
                #      100*np.abs(delta_outpt_power/1-delta_input_power) / delta_input_power)
                
                if   np.abs(delta_im3___power/3-delta_input_power) / delta_input_power < 0.1 and \
                     np.abs(delta_im2___power/2-delta_input_power) / delta_input_power < 0.1 and \
                     np.abs(delta_outpt_power/1-delta_input_power) / delta_input_power < 0.1:
                    iip2_dbm = input_power[-1] + (outpt_power[-1] - im2___power[-1])
                    oip3_dbm = outpt_power[-1] + (outpt_power[-1] - im3___power[-1])/2
                    # print(iip2, oip3)
            
                
                if oip3_dbm is not None and delta_outpt_power < delta_input_power-1:
                    # -- oip3_dbm not None so level is enoughly high
                    op1db_dbm = (outpt_power[-1] + outpt_power[-2]) / 2
                    # print(iip2, oip3)
            
            if power_dbm > 50:
                to_continue = False
            else:
                power_dbm += 2
            # -- END LOOP -----------------------------------------------------------

        if iip2_dbm is None:
            print("Erreur: impossible de caractériser finement les IIPx")
            iip2_dbm =  input_power[-1] + (outpt_power[-1] - im2___power[-1])
            oip3_dbm = (outpt_power[-1] +  outpt_power[-1] - im3___power[-1])/2

        if op1db_dbm is None:
            print("Erreur: impossible de caractériser finement l'OP1dB")
            op1db_dbm = (outpt_power[-1] + outpt_power[-2]) / 2

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.subplots(1, 1)
        ax1.axis('equal')
        for pwrs in (outpt_power, im2___power, im3___power):
            ax1.plot(input_power, pwrs)            
        ax1.set_xlabel('Input power (dBm)')
        ax1.set_ylabel('Output powers (dBm)')
        ax1.grid(True)
        plt.tight_layout()

        return op1db_dbm, iip2_dbm, oip3_dbm
# ------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------
class RF_channel(RF_Component):
    def __init__(self, rf_components=[]):
        self.rf_components = list(rf_components)

    def process_signals(self, signals, temp_kelvin=None):
        for rf_compnt in self.rf_components:
            rf_compnt.process(signals, temp_kelvin)
# ------------------------------------------------------------------------------------------
# ====================================================================================================
    
    
# ====================================================================================================
class Attenuator(RF_Component):
    def __init__(self, att_db, temp_kelvin=temperature_default):
        '''Attenuation is positive (for losses)'''
        self.temp_kelvin = temp_kelvin
        self.att_db      = att_db
        self.nf_db       = att_db
        
        self.gain        = gain_db_to_gain(-att_db)  # Convertir le gain de dB en linéaire
        self.nf          = nf_db_to_nf(self.nf_db)

    def process_signals(self, signals, temp_kelvin=None):
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin
        
        signals.sig2d += self.nf * Signals.generate_noise_dbm(signals.shape, thermal_noise_power_dbm(temp_kelvin, signals.bw_hz))
        signals.sig2d *= self.gain

ft  = lambda x,k: np.tanh(x/k)*k

class Simple_Amplifier(RF_Component):
    def __init__(self, gain_db, nf_db, op1db_dbm=20, oip3_dbm=10, iip2_dbm=40, temp_kelvin=temperature_default):
        self.temp_kelvin = temp_kelvin
        self.gain_db   = gain_db  
        self.nf_db     = nf_db
        self.op1db_dbm = op1db_dbm
        self.oip3_dbm  = oip3_dbm
        self.iip2_dbm  = iip2_dbm
        
        # -- Convertir  en linéaire
        self.gain  = gain_db_to_gain(self.gain_db)
        self.nf    = nf_db_to_nf(self.nf_db)
        
        self.iip2  = dbm_to_voltage(self.iip2_dbm )
        self.oip3  = dbm_to_voltage(self.oip3_dbm )
        self.op1db = dbm_to_voltage(self.op1db_dbm)

        self.a1 = self.gain
        self.a2 = 0.5 * self.a1 / self.iip2
        #self.a3 = 0.4 * self.a1/self.oip3**2
        
        self.k_oip3 = 0.24 * 10**(self.oip3_dbm/20)

    def process_signals(self, signals, temp_kelvin=None):
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin
        
        signals.sig2d += self.nf * Signals.generate_noise_dbm(signals.shape, thermal_noise_power_dbm(temp_kelvin, signals.bw_hz))
        signals.sig2d  = ft(self.a1 * signals.sig2d, self.k_oip3) + self.a2 * ft(signals.sig2d, self.op1db*2)**2
        
        # -- OP1dB : 1dB compression point output power
        signals.sig2d  = ft(signals.sig2d, self.op1db*6)

class RF_modelised_component(RF_Component):
    def __init__(self, freqs, gains_db, nfs_db, phases_rad=None, nominal_gain_for_im_db=None, op1db_dbm=np.inf, oip3_dbm=np.inf, iip2_dbm=np.inf, temp_kelvin=temperature_default):
        '''insertion_losses_dB is positive (for losses like an attenuation)'''
        self.temp_kelvin = temp_kelvin
        
        if freqs is None or gains_db is None or nfs_db is None:
            self.freqs      = None
            
            self.gains_db   = None
            self.gains      = None
            
            self.nfs_db     = None
            self.nfs        = None
            
            self.phases_rad = None
        else:
            self.freqs    = np.array(freqs)
            
            self.gains_db = np.array(gains_db)
            self.gains    = gain_db_to_gain(self.gains_db)
            
            self.nfs_db   = np.array(nfs_db)
            self.nfs      = nf_db_to_nf(self.nfs_db)
            
            if phases_rad is not None:
                self.phases_rad = np.array(phases_rad)
                self.gains      = self.gains * np.exp(1j * self.phases_rad)
        
        if nominal_gain_for_im_db is not None and nominal_gain_for_im_db > 0:
            self.nominal_gain_for_im_db = nominal_gain_for_im_db
            self.op1db_dbm = op1db_dbm
            self.oip3_dbm  = oip3_dbm
            if   iip2_dbm is not np.inf: self.iip2_dbm = iip2_dbm
            elif oip3_dbm is not np.inf: self.iip2_dbm = 40.
            else                       : self.iip2_dbm = np.inf
        elif nominal_gain_for_im_db is not None:
            print('[RF_modelised_component] Error: nominal_gain_for_im_db:', nominal_gain_for_im_db)
            nominal_gain_for_im_db = None
        
        if nominal_gain_for_im_db is None:
            self.nominal_gain_for_im_db = 1e-99
            self.op1db_dbm = np.inf
            self.oip3_dbm  = np.inf
            self.iip2_dbm  = np.inf

        self.iip2  = dbm_to_voltage(self.iip2_dbm )
        self.oip3  = dbm_to_voltage(self.oip3_dbm )
        self.op1db = dbm_to_voltage(self.op1db_dbm)
        
        # -- For distorsion
        self.a1 = gain_db_to_gain(self.nominal_gain_for_im_db)
        self.a2 = 0.5 * self.a1 / self.iip2
        #self.a3 = 0.4 * self.a1/self.oip3**2
        self.k_oip3 = 0.24 * 10**(self.oip3_dbm/20)
        
        # -- Initializing parameters to manage gains and losses out of band
        self.freqs_sup = 100e6 * np.arange(1, 11)
        self.freqs_inf = -self.freqs_sup[::-1]

        self.gains_sup_db = -5. * np.arange(1, 11) 
        self.gains_sup    = gain_db_to_gain(self.gains_sup_db)
        self.gains_inf    = self.gains_sup[::-1]
        
        self.nfs_sup = nf_db_to_nf(-self.gains_sup_db)
        self.nfs_inf = self.nfs_sup[::-1]
    
    def get_gains_nfs(self, signals, temp_kelvin=None):
        '''May be replaced by subclasses'''
        return self.freqs, self.gains, self.nfs
    
    def process_signals(self, signals, temp_kelvin=None):
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin
        
        # -- Get gains and nfs
        freqs, gains, nfs = self.get_gains_nfs(signals, temp_kelvin)
        #print(freqs)
        #print(gain_to_gain_db(gains))
        #print(nf_to_nf_db(nfs))

        # -- Extend the fequency domain with losses and noise figures        
        freqs = np.concat( (self.freqs_inf+freqs[0]        , freqs, self.freqs_sup+freqs[-1]        ) )

        gains_inf = self.gains_inf * np.exp( 1j * np.linspace(-np.angle(gains[ 0]), 0, num=len(self.gains_inf), endpoint=False)       )
        gains_sup = self.gains_sup * np.exp( 1j * np.linspace(-np.angle(gains[-1]), 0, num=len(self.gains_sup), endpoint=False)[::-1] )
        
        gains = np.concat( (gains_inf*gains[0]           , gains, gains_sup*gains[-1]           ) )
        nfs   = np.concat( (mul_nfs(self.nfs_inf, nfs[0]), nfs  , mul_nfs(self.nfs_sup, nfs[-1])) )
        
        # -- Extend to the negative frequencies
        gains = np.concat( (np.conjugate(gains[freqs>0][::-1]), gains[freqs>=0]) )
        nfs   = np.concat( (nfs[freqs>0][::-1]                , nfs[freqs>=0]  ) )
        freqs = np.concat( (-freqs[freqs>0][::-1]             , freqs[freqs>=0]) )

        #fig = plt.figure(figsize=(12, 6))
        #ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)
        #ax1.plot(freqs, gain_to_gain_db(gains)  )
        #ax2.plot(freqs, np.angle(gains))
        #ax3.plot(freqs, nf_to_nf_db(nfs))
        #ax3.set_xlabel('Freq')
        #ax1.grid(True)
        #plt.tight_layout()
        
        interp_gains = interp1d(freqs, gains, kind='linear', bounds_error=False, fill_value=(gains[0], gains[-1]))
        interp_nfs   = interp1d(freqs, nfs  , kind='linear', bounds_error=False, fill_value=(nfs[0]  , nfs[-1]  ))
        
        # -- Get spectrums of the signals
        spectrums = np.fft.fft(signals.sig2d, axis=1)
        fftfreqs  = np.fft.fftfreq(signals.n_points, 1/signals.sampling_rate)
        
        # -- Apply nf_gs (equivalent gains of NFs < 1)
        spectrums += interp_nfs(fftfreqs) * np.fft.fft( Signals.generate_noise_dbm(signals.shape, thermal_noise_power_dbm(temp_kelvin, signals.bw_hz)), axis=1 )
        spectrums *= interp_gains(fftfreqs)
        
        # -- Retrieve temporal signal
        signals.sig2d  = np.real( np.fft.ifft(spectrums, axis=1) )
        
        # -- Apply distorsion
        if not self.oip3 > 1e308:
            signals.sig2d /= self.a1
            signals.sig2d  = ft(self.a1 * signals.sig2d, self.k_oip3) + self.a2 * ft(signals.sig2d, self.op1db*2)**2
        
        if not self.op1db > 1e308:
            signals.sig2d = ft(signals.sig2d, self.op1db*6)

class RF_Cable(RF_modelised_component):
    def __init__(self, length_m, alpha=5.2e-06, insertion_losses_dB=0, temp_kelvin=temperature_default):
        '''insertion_losses_dB is positive (for losses like an attenuation)'''
        self.temp_kelvin         = temp_kelvin
        self.length_m            = length_m
        self.alpha               = alpha
        self.insertion_losses_dB = insertion_losses_dB
        
        super().__init__(None, None, None, temp_kelvin=temp_kelvin)
    
    def get_gains_nfs(self, signals, temp_kelvin=None):
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin

        freqs    = signals.freqs[signals.freqs>0]
        gains_db = -self.insertion_losses_dB - self.alpha * np.sqrt(freqs) * self.length_m
        gains    = gain_db_to_gain(gains_db)
        nfs      = nf_db_to_nf( -gains_db )

        #print(freqs)
        #print(gains_db)
        
        return freqs, gains, nfs

class HighPassFilter(RF_modelised_component):
    def __init__(self, cutoff_freq, order=1, q_factor=1, insertion_losses_dB=0, temp_kelvin=temperature_default):
        '''insertion_losses_dB is positive (for losses like an attenuation)'''
        self.temp_kelvin         = temp_kelvin
        self.cutoff_freq         = cutoff_freq
        self.order               = order
        self.q_factor            = q_factor
        self.insertion_losses_dB = insertion_losses_dB
        self.gain_losses         = 10**(-self.insertion_losses_dB / 20)  # insertion_losses_dB as a negative gain
        
        super().__init__(None, None, None, temp_kelvin=temp_kelvin)
    
    def get_gains_nfs(self, signals, temp_kelvin=None):
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin
        
        freqs = signals.freqs[signals.freqs>0]
        b, a  = butter(self.order, self.cutoff_freq, btype='high', fs=signals.sampling_rate, output='ba' )
        w, h  = freqz(b, a, worN=freqs, fs=signals.sampling_rate)
        
        gains    = h
        gains_db = np.minimum(0., gain_to_gain_db(gains))
        nfs      = nf_db_to_nf( -gains_db )
        
        return freqs, gains, nfs

class Antenna_component(RF_Component):
    def __init__(self, freqs, gains_db, phases_rad=None, temp_kelvin=temperature_default):
        '''insertion_losses_dB is positive (for losses like an attenuation)'''
        self.temp_kelvin = temp_kelvin
        
        self.freqs       = np.array(freqs)
        
        self.gains_db    = np.array(gains_db)
        self.gains       = gain_db_to_gain(self.gains_db)
        
        if phases_rad is not None:
            self.phases_rad = np.array(phases_rad)
            self.gains      = self.gains * np.exp(1j * self.phases_rad)
        
        # -- Initializing parameters to manage gains and losses out of band
        self.freqs_sup = 100e6 * np.arange(1, 11)
        self.freqs_inf = -self.freqs_sup[::-1]

        self.gains_sup_db = -5. * np.arange(1, 11) 
        self.gains_sup    = gain_db_to_gain(self.gains_sup_db)
        self.gains_inf    = self.gains_sup[::-1]
        

        # -- Extend the fequency domain with losses and noise figures        
        self.freqs = np.concat( (self.freqs_inf+freqs[0]        , freqs, self.freqs_sup+freqs[-1]        ) )

        self.gains_inf = self.gains_inf * np.exp( 1j * np.linspace(-np.angle(gains[ 0]), 0, num=len(self.gains_inf), endpoint=False)       )
        self.gains_sup = self.gains_sup * np.exp( 1j * np.linspace(-np.angle(gains[-1]), 0, num=len(self.gains_sup), endpoint=False)[::-1] )
        
        self.gains = np.concat( (self.gains_inf*self.gains[0]           , self.gains, self.gains_sup*self.gains[-1]           ) )
        
        # -- Extend to the negative frequencies
        self.gains = np.concat( (np.conjugate(gains[freqs>0][::-1]), gains[freqs>=0]) )
        self.freqs = np.concat( (-freqs[freqs>0][::-1]             , freqs[freqs>=0]) )
    
    def process_signals(self, signals, temp_kelvin=None):
        temp_kelvin = temp_kelvin if temp_kelvin else self.temp_kelvin
        
        interp_gains = interp1d(self.freqs, gains, kind='linear', bounds_error=False, fill_value=(gains[0], gains[-1]))
        
        # -- Get spectrums of the signals
        spectrums  = np.fft.fft(signals.sig2d, axis=1)
        fftfreqs   = np.fft.fftfreq(signals.n_points, 1/signals.sampling_rate)
        
        spectrums *= interp_gains(fftfreqs)
        
        # -- Retrieve temporal signal
        signals.sig2d  = np.real( np.fft.ifft(spectrums, axis=1) )
        signals.sig2d += Signals.generate_noise_dbm(signals.shape, thermal_noise_power_dbm(temp_kelvin, signals.bw_hz))
# ====================================================================================================


# ====================================================================================================
if __name__ == '__main__':
    signal = Signals(10e-9, 40e9)
    signal.add_noise( thermal_noise_power_dbm(signal.temp_kelvin, signal.bw_hz) )
    signal.add_tone( 3e9,   0,   0  )
    signal.add_tone(11e9, -55,  pi/4)
    signal.add_tone(17e9, -50, -pi/3)
    
    print(signal.rms_dbm())
    signal.plot_temporal(10e-9)
    signal.plot_spectrum()
    
    # Création d'un amplificateur avec non-linéarités
    amplifier = Amplifier(gain_db=20, iip2_dbm=30, oip3_dbm=20, nf_db=5)
    
    # Traitement du signal par l'amplificateur
    signal.signal = amplifier.process(signal)
    
    print(signal.rms_dbm())
    signal.plot_temporal(10e-9)
    signal.plot_spectrum()
        
    # Création d'un filtre passe-haut non idéal
    high_pass_filter = HighPassFilter(cutoff_freq=6e9, order=5, q_factor=0.7)
    
    # Application du filtre passe-haut au signal amplifié
    signal.signal = high_pass_filter.process(signal)
    
    print(signal.rms_dbm())
    signal.plot_temporal(10e-9)
    signal.plot_spectrum()
# ====================================================================================================
