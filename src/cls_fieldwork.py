#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date    : 2023-12-07
@Author  : Jakob Christiaanse (j.c.christiaanse@tudelft.nl)

coastal-fieldwork source code. This script contains classes and functions to read and process coastal fieldwork data.

"""

#%% Import
import lzma
import pickle
import re
from datetime import timedelta, datetime
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyrsktools import RSK
from scipy.fft import fft
from scipy.signal import detrend
from tqdm import tqdm

pd.options.mode.copy_on_write = True


# Class for storing instrument metadata (ID, frequency, station, gps locations, etc.)
class Instrument:
    """
    A class for storing and managing metadata for instruments.

    Attributes:
        id (str): The instrument's unique identifier.
        type (str): The type of instrument (e.g., 'hobo', 'solo').
        freq (str, optional): The measurement frequency of the instrument (e.g., '1Hz').
        dep (str, optional): Identifier of the deployment in which the instrument was deployed.
        station (str, optional): The station where the instrument was deployed.
        positions (list, optional): GPS or physical positions associated with the instrument.
        ufreq (str, optional): The frequency of the instrument in microsecond intervals if applicable.
    """

    def __init__(self, inst_id, inst_type, freq=None, dep=None, station=None, positions=None):
        """
        Initialize the Instrument object.

        Parameters:
            inst_id (str): The unique identifier for the instrument.
            inst_type (str): The type of instrument (e.g., 'hobo', 'solo', 'rbr').
            freq (str, optional): The measurement frequency (e.g., '1Hz'). If the frequency is given in Hz,
                                  it will be converted to a microsecond interval stored in the 'ufreq' attribute.
            dep (str, optional): The deployment identifier.
            station (str, optional): The station identifier where the instrument is located.
            positions (list, optional): A list of positions (e.g., GPS coordinates) for the instrument.

        Raises:
            ValueError: If the frequency is provided in an unexpected format or cannot be processed.
        """
        self.id = inst_id  # instrument ID
        self.type = inst_type  # instrument type (hobo, solo, spot)
        self.freq = freq  # measurement frequency
        self.dep = dep  # deployment ID
        self.station = station  # measurement station ID
        self.positions = positions  # measured instrument positions

        # If the frequency is in Hz, it can't be used for resampling, so we add another attribute
        # 'ufreq' (unit frequency), transforming Hz to microsecond intervals (denoted by 'us')
        if self.freq is not None and 'Hz' in self.freq:
            hz = re.findall(r'\d+', self.freq)
            if hz:
                self.ufreq = '%dus' % (1 / int(hz[0]) * 10e5)
            else:
                raise ValueError(f"Unable to extract frequency from '{self.freq}'")
        else:
            self.ufreq = self.freq

    def __str__(self):
        """
        Return a string representation of the Instrument object.

        Returns:
            str: A string describing the instrument, including ID, type, frequency, and station.
        """
        return 'Instrument: %s, Type: %s, Frequency: %s, Station: %s' % (self.id, self.type, self.freq, self.station)

    def __repr__(self):
        """
        Return the official string representation of the Instrument object.
        Calls the __str__ method.

        Returns:
            str: The string representation of the object.
        """
        return self.__str__()

    def __getstate__(self):
        """
        Return the state of the Instrument object for pickling.

        If the instrument type is 'rbr', 'solo', or 'rsk', certain dataframes will be excluded
        from the state to reduce file size.

        Returns:
            dict: A dictionary containing the object's state, excluding specific dataframes for certain instrument types.
        """
        d = self.__dict__
        if self.type.lower() in ['rbr', 'solo', 'rsk']:
            self_dict = {k: d[k] for k in d if k not in ['df_raw', 'df_proc', 'df_16Hz', 'df_2Hz', 'df_1S']}
        else:
            self_dict = {k: d[k] for k in d}
        return self_dict

    def __setstate__(self, state):
        """
        Restore the state of the Instrument object from a pickled state.

        Parameters:
            state (dict): The pickled state to restore.
        """
        self.__dict__ = state

    def add_positions(self, positions):
        """
        Add positions (e.g., GPS coordinates) to the instrument.

        Parameters:
            positions (list): A list of positions to associate with the instrument.
        """
        self.positions = positions

    def save_obj(self, outpath):
        """
        Save the current state of the Instrument object to a file using pickle.

        Parameters:
            outpath (str): The output file path where the object will be saved.
        """
        with open(outpath, 'wb') as f:
            pickle.dump(self, f)


# Child class to Instrument for processing Pressure logger data
class PL(Instrument):
    """
    A class for processing Pressure Logger (PL) data, extending the Instrument class.

    The PL class is designed to manage and process data from pressure loggers (RBR Solo or HOBO U20L).
    It provides functionality to read raw data, apply offsets, correct atmospheric pressure, calculate
    water depth and water level, and perform time-series analysis. This class also allows saving/loading
    processed data and exporting results for further analysis.

    Attributes:
        keys (list): List of keys for all linked DataFrames (e.g., 'raw', 'proc').
        rd (str): Instrument retrieval date.
        dir (str): Path to the working data directory.
        p_atm (float or None): Atmospheric pressure (default is None).
        p_offset (float or None): Pressure offset applied to the data (default is None).
        d_offset (float or None): Water depth offset (default is None).
        df_proc (pd.DataFrame or None): DataFrame containing processed data.
        time_activity (list or None): List of activity times (used for data cleaning).
        ufac_p (dict): Conversion factors for pressure (base = mbar).
        ufac_h (dict): Conversion factors for depth (base = meters).

    Methods:
        __getstate__: Prepare the object's state for pickling, excluding DataFrames to reduce file size.
        read_df: Load raw or processed data from pickled files into the PL object.
        time_offset: Adjust the timestamps of the data by a specified time offset.
        pressure_offset: Apply a logger-specific pressure offset and correct the pressure data.
        atm_pres: Apply atmospheric pressure correction to the processed data.
        derive_depth: Calculate water depth from the corrected pressure using hydrostatic or TEOS-10 methods.
        derive_wl: Calculate water level by applying instrument elevation and depth offsets.
        wl_offset: Adjust the water level by adding a specified offset during a given time frame.
        clean_ts: Insert NaN values into time intervals for cleaning data, with optional resampling.
        resample: Resample the DataFrame at a specified frequency and with a specified aggregation function.
        plot_ts: Plot time series data for a specific variable, optionally resampling or applying time frames.
        plot_bucket: Plot bucket test data for visualizing water level and instrument measurements.
        save_obj: Save the current state of the PL object (excluding DataFrames) to a file.
        save_df: Save specified DataFrames (raw or processed) to a file in various formats.
    """
    # Conversion factors for pressure (base = mbar)
    ufac_p = {
        'Pa': 0.01,
        'mbar': 1,
        'kPa': 10,
        'dbar': 100,
        'bar': 1000,
    }

    # Conversion factors for depth (base = meters)
    ufac_h = {
        'mm': 0.001,
        'cm': 0.01,
        'dm': 0.1,
        'm': 1,
        'km': 1000,
    }

    ##### INITIALIZATION
    def __init__(self, fnames, inst_id, inst_type='solo', freq='2Hz', unit='mbar', station=None, channels=['pressure']):
        """
        Initialize a presuure logger (PL) object by reading raw data from the specified file(s), extracting relevant
        information, and setting up necessary attributes. Currently, the following PL types adn raw files are supported:
            RBR Solo, *.rsk files
            HOBO U20L, *.csv files (the raw *.hobo files can be exported as CSV from HoboWare)

        Parameters:
            fnames (str or list): File name(s)/path(s) of the data files to read.
            inst_id (str): Instrument ID.
            inst_type (str): Type of instrument, options are 'hobo', 'rbr', 'solo'.
            freq (str): Measuring frequency (default is '2Hz').
            unit (str): Unit of the raw pressure data. Can be one of ['Pa', 'kPa', 'mbar', 'dbar', 'bar'].
                        The pressure will be converted to 'mbar'.
            station (str, optional): Station ID where the logger was deployed.
            channels (list, optional): Channels to read for Hobos, options are 'pressure' and/or 'temperature'.

        Raises:
            ValueError: If the file format or instrument type is not recognized.
        """

        # Initialize attributes
        super().__init__(inst_id, inst_type, freq, station=station)
        self.keys = ['raw']  # list to store the keys of all linked DataFrames

        # Read & store retrieval date and data directory
        if not isinstance(fnames, list):
            fnames = [fnames]
        self.rd = fnames[-1].split('/')[-1].split('_')[0]  # instrument retrieval date
        self.dir = join('/'.join(fnames[0].split('/')[:-3]), 'data')  # working directory

        # Initialize attributes to be defined later
        self.p_atm = None  # Atmospheric pressure
        self.p_offset = None  # Logger pressure offset
        self.d_offset = None  # Water depth offset
        self.df_proc = None  # DataFrame for processed data
        self.time_activity = None  # Instrument activity times

        # Handle HOBO file(s) (.csv)
        if inst_type.lower() in ['hobo']:
            self.D = 3  # instrument diameter in cm
            self.channels = channels  # channels to be used
            fill_method = None

            csv = {}
            for i, f in enumerate(fnames):
                csv[i] = (pd.read_csv(f, header=1, usecols=list(np.arange(0, len(channels) + 2)), index_col=0)
                          .set_axis(['datetime'] + channels, axis=1)
                          .set_index('datetime', drop=True)
                          )
                if len(fnames) > 1:
                    csv[i].index = pd.to_datetime(csv[i].index, format='mixed', dayfirst=True)
                else:
                    csv[i].index = pd.to_datetime(csv[i].index, format='%d/%m/%y %H:%M:%S')

            # Concatenate individual dfs into one timeseries
            if len(fnames) > 1:
                self.df_raw = pd.concat([csv[key] for key in csv]).sort_index()
            else:
                self.df_raw = csv[0]

        # Handle RBR Solo file (.rsk)
        elif inst_type.lower() in ['rbr', 'solo']:
            self.D = 2  # instrument diameter in cm
            fill_method = 'nearest'

            # Open and read RSK file with data
            with RSK(join(fnames[0])) as rsk:
                rsk.readdata()

                # Define properties & parameters
                self.serial = rsk.instrument.serialID
                self.parameters = {
                    'temparature': rsk.parameterKeys[9].value,
                    'pressure': rsk.parameterKeys[8].value,
                    'atm_pressure': rsk.parameterKeys[-6].value,
                    'density': rsk.parameterKeys[5].value,
                    'salinity': rsk.parameterKeys[13].value,
                    'conductivity': rsk.parameterKeys[-5].value,
                }

                # Store timeseries as df_raw
                self.df_raw = pd.DataFrame(rsk.data).set_index('timestamp')
                rsk.close()

        else:
            raise ValueError(f"Unsupported instrument type: {inst_type}")

        # Reindex to get consistent timeseries (gaps filled with NaNs) and convert pressure to mbar
        self.ts = self.df_raw.index[0]
        self.te = self.df_raw.index[-1]
        date_range = pd.date_range(start=self.ts, end=self.te, freq=self.ufreq)
        self.df_raw = self.df_raw.reindex(date_range, method=fill_method)
        self.df_raw.pressure = self.df_raw.pressure * self.ufac_p[unit]

    def __getstate__(self):
        """
        Get the state of the PL object for pickling, excluding any dataframes for reduced file size.

        Returns:
            dict: A dictionary containing the object's state.
        """
        d = self.__dict__
        self_dict = {k: d[k] for k in d if 'df' not in k}
        return self_dict

    def read_df(self, dfs='all', ext='parq'):
        """
        Load data into the PL object. Data should be in the form of a pickled pandas DF, inside the working directory
        and named as 'YYMMDD_SSS_INSTID_df_KEY.ext', where YYMMDD is the retrieval date of the insturment,
        SSS is the station, INSTID is the unique identifier, and KEY is the DataFrame key.

        Parameters:
            dfs (str or list of str): Specifies which DataFrame(s) to load. Options are 'raw', 'proc', 'all',
                                      or specific resolution keys.
            ext (str): The file extension of the saved data ('parq' for parquet, 'p' for pickle).

        Raises:
            AttributeError: If a requested DataFrame key is not found.
        """
        if dfs == 'all':
            dfs = self.keys
        elif not isinstance(dfs, list):
            dfs = [dfs]

        for df in dfs:
            if ext in ['parq', 'parquet']:
                self.__setattr__('df_%s' % df, pd.read_parquet(
                    join(self.dir, '%s_%s_%s_df_%s.%s' % (self.rd, self.station, self.id, df, ext))))
            else:
                self.__setattr__('df_%s' % df, pd.read_pickle(
                    join(self.dir, '%s_%s_%s_df_%s.%s' % (self.rd, self.station, self.id, df, ext))))

    ##### DATA PROCESSING
    def time_offset(self, hours=0, minutes=0, seconds=0, df_keys=None):
        """
        Adjust the timestamps of specified DataFrames by adding a specified time offset.

        This method adds a `timedelta` to the timestamps of the selected DataFrames. For subtracting time,
        negative values can be passed for `hours`, `minutes`, or `seconds`.

        Parameters:
            hours (int, optional): Number of hours to add to the timeseries. Default is 0.
            minutes (int, optional): Number of minutes to add to the timeseries. Default is 0.
            seconds (int, optional): Number of seconds to add to the timeseries. Default is 0.
            df_keys (list of str, optional): List of DataFrame keys (e.g., 'raw', 'proc') to apply the offset to.
                                             If None, defaults to ['raw'].

        Raises:
            AttributeError: If a specified DataFrame key is not found in the object.
        """
        if df_keys is None:
            df_keys = ['raw']

        # Create a timedelta object from the provided hours, minutes, and seconds
        time_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)

        # Apply the time offset to each specified DataFrame
        for key in df_keys:
            try:
                # Adjust the index of the DataFrame by the computed timedelta
                self.__getattribute__('df_%s' % key).index += time_delta
            except AttributeError:
                raise AttributeError(f"DataFrame with key '{key}' not found.")

    def pressure_offset(self, p_offset=0, unit='mbar', interp='step'):
        """
        Correct the pressure data by applying a logger offset and create a processed DataFrame.

        Parameters:
            p_offset (int, float, pd.Series, pd.DataFrame, list): The pressure offset to apply.
            unit (str): Unit of the pressure offset. Can be one of ['Pa', 'kPa', 'mbar', 'dbar', 'bar']. The offset
                        will be converted to 'mbar' (default).
            interp (str): Interpolation method for applying the offset if time-based offsets are provided.
                          Options are 'step' (default) or 'linear'.

        Raises:
            ValueError: If an invalid interpolation method is provided.
            TypeError: If the `p_offset` type is not recognized (should be int, float, pd.Series, or pd.DataFrame).
        """
        if isinstance(p_offset, (float, int)):
            self.p_offset = p_offset * self.ufac_p[unit]

        elif isinstance(p_offset, (pd.Series, pd.DataFrame)):
            p_offset = p_offset.set_axis(['p_offset'], axis=1) * self.ufac_p[unit]

            # If there is only one p_offset value, use it for the entire time series
            if len(p_offset.p_offset.unique()) == 1:
                p_offset = p_offset.iloc[0].p_offset
                self.p_offset = p_offset

            # If there are multiple p_offset values at specific timesteps, it depends on the interpolation method
            else:
                self.p_offset = p_offset.p_offset.mean()
                # For step, each p_offset value is used from its corresponding timestamp until the next timestamp
                if interp == 'step':
                    p_offset = (p_offset.reindex(pd.date_range(self.ts, self.te, freq=self.ufreq))
                                        .ffill()
                                        .bfill()
                                        .to_numpy()
                                )
                # For linear, p_offset values are linearly interpolated through the timeseries
                elif interp == 'linear':
                    p_offset = (p_offset.reindex(pd.date_range(self.ts, self.te, freq=self.ufreq))
                                        .interpolate(method='time', limit_direction='both')
                                        .to_numpy()
                                )
                else:
                    raise ValueError("Invalid interpolation method. Choose 'step' or 'linear'.")

        else:
            raise TypeError("p_offset must be int, float, pd.Series, or pd.DataFrame.")

        # Create dataframe with corrected sea pressure (rename pressure to p)
        self.df_proc = (self.df_raw.copy() - p_offset).rename({'pressure': 'p'}, axis=1)
        self.keys.append('proc')

    def atm_pres(self, p_atm=1013.25, unit='mbar'):
        """
        Apply atmospheric pressure correction to the pressure data.

        Parameters:
            p_atm (int, float, pd.Series, pd.DataFrame): The atmospheric pressure to subtract. Can be a constant
                value or a time series.
            unit (str): The unit of atmospheric pressure, default is 'mbar'. Can be one of ['Pa', 'kPa', 'mbar', 'dbar',
                'bar'].

        Raises:
            TypeError: If the type of `p_atm` is not recognized (should be int, float, pd.Series, or pd.DataFrame).
        """
        # If processed dataframe is not yet created, do it now
        if 'proc' not in self.keys:
            self.df_proc = self.df_raw.copy().rename({'pressure': 'p'}, axis=1)
            self.keys.append('proc')

        # For a constant p_atm value, simply subtract it from the entire pressure timeseries
        if isinstance(p_atm, (float, int)):
            self.p_atm = p_atm * self.ufac_p[unit]
            self.df_proc.p = self.df_proc.p - self.p_atm

        # For a timeseries of p_atm values, subtract element-wise
        elif isinstance(p_atm, (pd.Series, pd.DataFrame)):
            p_atm = (p_atm.set_axis(['p'], axis=1)
                     .reindex(pd.date_range(self.ts, self.te, freq=self.ufreq), method='nearest')
                     ) * self.ufac_p[unit]
            self.df_proc.p = self.df_proc.p - p_atm.p

        else:
            raise TypeError("p_atm must be a float, int, pd.Series, or pd.DataFrame.")

    def derive_depth(self, method='hydrostatic', rho=1023.6, g=9.81, lat=45):
        """
        Derive the water depth from the corrected pressure. Can either be done through a simple hydrostatic
        calculation (d = P/rho*g) or using the TEOS-10 library which takes into account latitudinal changes among
        others (this is what Ruskin software uses). Only use this method after having corrected the raw pressure for
        atmospheric pressure using atm_pres()

        Parameters:
            method (str): Method for depth calculation. 'hydrostatic' (default) or 'teos-10'/'gsw'.
            rho (float): Water density in kg/m³. Default is 1023.6 kg/m³ (seawater).
            g (float): Gravitational acceleration in m/s². Default is 9.81 m/s².
            lat (float): Latitude for TEOS-10 calculations. Default is 45 degrees.

        Raises:
            ValueError: If an invalid method is provided.
        """
        # If processed dataframe is not yet created then there has not been an atmospheric pressure correction,
        # which should be done first
        if 'proc' not in self.keys:
            raise ValueError('No proccesed dataframe found. Make sure to first correct the raw data for atmospheric '
                             'pressure using atm_pres()')
        else:
            if method == 'hydrostatic':
                self.df_proc['d'] = self.df_proc.p * 100 / (rho * g)
            elif method in ['teos', 'teos-10', 'gsw']:
                import gsw
                self.df_proc['d'] = -gsw.z_from_p(self.df_proc.p / 100, lat)
            else:
                raise ValueError("Invalid method. Choose 'hydrostatic' or 'teos-10'.")

    def derive_wl(self, z=0, d_offset=0, interp='step', unit='m'):
        """
        Derive a referenced water level (wl) by applying instrument elevation and depth offsets. This method can only be
        run when the water depth has been derived using derive_depth().

        Parameters:
            z (float or pd.DataFrame): Elevation of the instrument. Can be a constant or a time series.
            d_offset (float): Optional depth offset. Default is 0.
            interp (str): Interpolation method for multiple elevation values. Options are 'step' (default) or 'linear'.
            unit (str): Unit for the elevation. Default is 'm'. Supported units are 'mm', 'cm', 'm'.

        Raises:
            TypeError: If the `z` parameter type is not recognized.
            ValueError: If the interpolation method is invalid.
        """
        if 'proc' not in self.keys:
            raise ValueError('No proccesed dataframe found. Make sure to first correct the raw data for atmospheric '
                             'pressure using atm_pres(), and derive the water depth using derive_depth().')
        else:
            self.d_offset = d_offset * self.ufac_h[unit]

            if isinstance(z, (float, int)):
                self.df_proc['z'] = z * self.ufac_h[unit]

            elif isinstance(z, (pd.Series, pd.DataFrame)):
                z = z.set_axis(['z'], axis=1) * self.ufac_h[unit]

                # If there is only one z-value, use it for the entire time series
                if len(z.z.unique()) == 1:
                    self.df_proc['z'] = z.iloc[0].z

                # If there are multiple z-values at specific timesteps, it depends on the interpolation method
                else:
                    # For step, each z value is used from its corresponding timestamp up until the next timestamp
                    if interp == 'step':
                        self.df_proc['z'] = z.reindex(pd.date_range(self.ts, self.te, freq=self.ufreq)).ffill().bfill()
                    # For linear, z-values are interpolated through the timeseries
                    elif interp == 'linear':
                        self.df_proc['z'] = (z.reindex(pd.date_range(self.ts, self.te, freq=self.ufreq))
                                             .interpolate(method='time', limit_direction='both'))
                    else:
                        raise ValueError("Invalid interpolation method. Choose 'step' or 'linear'.")

            else:
                raise TypeError("z must be a float, int, pd.Series, or pd.DataFrame.")

            # compute water level at each timestamp
            self.df_proc['h'] = self.df_proc.d + self.df_proc.z + self.d_offset

    def wl_offset(self, delta_h, time_frame):
        """
        Adjust the derived water level by adding a specified offset during a time frame.

        Parameters:
            delta_h (float): The offset to add to the water level.
            time_frame (list of two datetime objects):  The time range during which the offset should be applied.
                                                        Should be a list with two elements: [start_time, end_time].
        """
        self.df_proc.loc[time_frame[0]:time_frame[1], 'h'] += delta_h

    def clean_ts(self, times=None, resample=1, save=0):
        """
        Clean the timeseries data by inserting NaNs for the specified time intervals. The clean data TS will be stored
        as additional dataframe with a key that is equal to the frequency of the instrument (e.g., df_2Hz).

        Parameters:
            times (list or pd.Series): Time intervals to be cleaned. If it's a list, each element should be a tuple with
                                       two elements: start and end times. If it's a pandas Series, the indices where the
                                       values are 0 will be cleaned.
            resample (bool, optional): Whether to resample the provided times to the same frequency as the DataFrame.
                                       Default is True.
            save (bool, optional):     Whether to save the cleaned DataFrame. Default is False.

        Raises:
            TypeError: If the `times` parameter type is not recognized (should be a list of tuples or a pandas Series).

        """
        # Handle time intervals for cleaning
        if times is None:
            times = []

        # Initialize a copy of the DataFrame to clean
        if self.freq == '16Hz':
            df_clean = self.df_proc.copy().resample(self.ufreq).nearest()
        else:
            df_clean = self.df_proc.copy()

        # Store times and resample to same frequency as df, if required
        self.time_activity = times.copy()
        if resample:
            times = times.resample(self.ufreq).nearest()

        # clean by inserting nans
        if isinstance(times, list):
            for t in times:
                df_clean.loc[t[0]:t[1]] = np.nan
        elif isinstance(times, pd.Series):
            df_clean.loc[df_clean.index.isin(times[times == 0].index)] = np.nan
        elif isinstance(times, pd.DataFrame):
            df_clean.loc[df_clean.index.isin(times[times[self.id] == 0].index)] = np.nan
        else:
            raise TypeError("'times' must be a list of tuples or a pandas Series.")

        # Set the cleaned DataFrame as an attribute and update the keys list
        self.__setattr__('df_%s' % self.freq, df_clean)
        self.keys.append(self.freq)

        # Optionally save the cleaned DataFrame to parquet file
        if save:
            self.save_df(dfs='%s' % self.freq, ext='parq')

    def resample(self, freq, base=None, agg_func='mean', save=0):
        """
        Resamples the DataFrame with a specified frequency and aggregation function.

        Parameters:
            freq (str or timedelta): The frequency for resampling (e.g., '1H' for 1 hour).
                                     For frequencies in Hz, it will automatically convert to microseconds.
            base (str, optional): The DataFrame key to resample from. Default is None, which uses the main frequency.
            agg_func (str, optional): The aggregation function to apply. Default is 'mean'.
                                      Supported functions: 'mean', 'median', 'sum', 'min', 'max', 'nearest'.
            save (bool, optional): Whether to save the resampled DataFrame. Default is False.

        Raises:
            ValueError: If an invalid aggregation function is provided.
        """
        # If a frequency is given in Hz, first convert it to microseconds
        if 'hz' in freq.lower():
            hz = re.findall(r'\d+', freq)
            ufreq = '%dus' % (1 / int(hz[0]) * 10e5)
        else:
            ufreq = freq

        # Determine base DF from which to resample
        if not base:
            base_df = 'df_%s' % self.freq
        else:
            base_df = f'df_{base}' if 'df' not in base else base

        # Resample timeseries according to aggregation function
        if agg_func == 'mean':
            resampled = self.__getattribute__(base_df).resample(ufreq).mean()
        elif agg_func == 'median':
            resampled = self.__getattribute__(base_df).resample(ufreq).median()
        elif agg_func == 'sum':
            resampled = self.__getattribute__(base_df).resample(ufreq).sum()
        elif agg_func == 'min':
            resampled = self.__getattribute__(base_df).resample(ufreq).min()
        elif agg_func == 'max':
            resampled = self.__getattribute__(base_df).resample(ufreq).max()
        elif agg_func == 'nearest':
            resampled = self.__getattribute__(base_df).resample(ufreq).nearest()
        else:
            raise ValueError("Invalid aggregation function.")

        # Store the resampled DataFrame with a new key
        self.__setattr__('df_%s' % freq, resampled)
        self.keys.append(freq)

        # Save the resampled DataFrame to parquet file
        if save:
            self.save_df(dfs='%s' % freq, ext='parq')

    ##### PLOTTING
    def plot_ts(self, df=None, var='h', time_frame=None, ax=None, resample=None, hlines=None, show=1, **kwargs):
        """
        Plot time series data for a specified variable within a given time frame.

        Parameters:
            df (str, optional): The DataFrame key to plot. Default is the current frequency DataFrame.
            var (str): The variable to plot (e.g., 'h' for water level). Default is 'h'.
            time_frame (list or tuple, optional): Time range for the plot [start_time, end_time]. Default is full range.
            ax (matplotlib axis, optional): The axis to plot on. Default is None (creates a new plot).
            resample (str, optional): The resampling frequency for the plot. Default is None.
            hlines (list, optional): List of y-values for horizontal reference lines. Default is None.
            show (bool, optional): Whether to display the plot. Default is True.

        Raises:
            ValueError: If the specified DataFrame key or resample frequency is not found.
        """
        # If no time frame is specified, plot entire time series
        if time_frame is None:
            time_frame = [self.ts, self.te]

        # If no df is specified, use clean DF with base frequency
        if df is None:
            df = 'df_%s' % self.freq

        # Plot resampled data?
        if resample is None:
            plot_df = self.__getattribute__(df).loc[time_frame[0]:time_frame[1], var]
        elif resample in [el for el in self.keys]:
            plot_df = self.__getattribute__('df_%s' % resample).loc[time_frame[0]:time_frame[1], var]
        else:
            plot_df = self.__getattribute__(df).loc[time_frame[0]:time_frame[1], var].resample(resample).mean()

        # Create a new plot if no axis is provided
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))

        plot_df.plot(ax=ax, label=var, **kwargs)

        # Add horizontal lines for reference if provided
        if hlines:
            ax.hlines(hlines, time_frame[0], time_frame[1], ls=':', color='C3')

        ax.grid()
        ax.set(ylabel=var, xlim=(time_frame[0], time_frame[1]))
        ax.legend()
        plt.tight_layout()

        if show:
            plt.show()

    def plot_bucket(self, h, time_frame=None, resample=None, ax=None, show=False):
        """
        Plot data for a bucket test, showing water depth and bucket level.

        Parameters:
            h (float): Water level in the bucket (in cm).
            time_frame (list or tuple, optional): Time range for the plot [start_time, end_time]. Default is None.
            resample (str, optional): Resampling frequency for the plot. Default is None.
            ax (matplotlib axis, optional): Axis to plot on. Default is None (creates a new plot).
            show (bool, optional): Whether to display the plot. Default is False.
        """
        self.plot_ts(df='df_proc', var='d', time_frame=time_frame, ax=ax, hlines=[(h - 0.5 * self.D) / 100],
                     resample=resample, show=show)

    ##### SAVING
    def save_obj(self, ext='p'):
        """
        Save the PL object (excluding DataFrames) to a file using pickle or lzma compression.

        Parameters:
            ext (str): File extension for saving. Options are 'p' for pickle and 'xz' for lzma-compressed pickle. Default is 'p'.
        """
        fname = '%s_%s_%s.%s' % (self.rd, self.station, self.id, ext)

        if ext == 'p':
            with open(join(self.dir, fname), 'wb') as f:
                pickle.dump(self, f)
        else:
            with lzma.open(join(self.dir, fname), 'wb') as f:
                pickle.dump(self, f)

    def save_df(self, dfs='raw', ext='parq'):
        """
        Save the specified DataFrame(s) to a file.

        Parameters:
            dfs (str or list of str): Specifies which DataFrame(s) to save. Use 'all' to save all associated DataFrames.
                                      If a list is provided, it should contain specific DataFrame keys.
            ext (str): File extension for saving. Supported formats: 'parq' (Parquet, default), 'xz', 'gz', 'p'.

        Raises:
            AttributeError: If the specified DataFrame does not exist.
        """
        if dfs == 'all':
            dfs = self.keys
        elif not isinstance(dfs, list):
            dfs = [dfs]

        for df in dfs:
            fname = '%s_%s_%s_df_%s.%s' % (self.rd, self.station, self.id, df, ext)
            print('Saving df_%s to %s' % (df, fname))
            try:
                if ext == 'parq':
                    self.__getattribute__('df_%s' % df).to_parquet(join(self.dir, fname))
                else:
                    self.__getattribute__('df_%s' % df).to_pickle(join(self.dir, fname), compression='infer')

            except AttributeError as e:
                print(e, f'DataFrame df_{df} does not exist for {self.type}')


# Child class to Instrument for processing Spotter wave buoy data
class SPOT(Instrument):
    """
    A class for processing Spotter wave buoy data, extending the Instrument class.

    The SPOT class is designed to handle the data collected from Spotter wave buoys,
    including raw and bulk displacement data, sea surface temperature, wave spectra,
    and other related measurements. It provides methods to load, process, and analyze
    the data, as well as visualize and export results.

    Attributes:
        dir (str): Path to the Spotter data directory.
        rdir (str): Path to the raw data directory.
        bdir (str): Path to the bulk data directory.
        dpl_raw (pd.DataFrame or None): DataFrame containing raw displacement data.
        dpl (pd.DataFrame or None): DataFrame containing processed displacement data.
        spectra (pd.DataFrame or None): DataFrame containing wave spectra data.
        ts (datetime or None): Start time of the instrument activity.
        te (datetime or None): End time of the instrument activity.
        vars (dict): Dictionary mapping variable names (e.g., 'dpl', 'sst') to data categories and columns.
        bulk_vars (dict): Dictionary of bulk data parameters (e.g., 'Hs', 'Tp') and their units.
        bulk_df (pd.DataFrame): DataFrame containing bulk measurement data (e.g., wave and wind parameters).

    Methods:
        __getstate__: Prepare the object's state for pickling, excluding displacement data to reduce file size.
        read_df: Load raw or processed data from pickled files into the SPOT object.
        time_offset: Adjust the timestamps of the data by a specified time offset.
        active_dates: Set the active date range for the analysis by filtering the data to the specified time range.
        skip_dates: Exclude data within specified time ranges, with an optional interpolation to fill gaps.
        compute_spectra: Compute wave energy spectra from displacement data over a given time range.
        plot_ts: Plot a time series for a specific variable within a time range, with options for resampling.
        save_obj: Save the current state of the SPOT object (excluding DataFrames) to a file.
        save_df: Save specific DataFrames (raw or processed) to a file in various formats.
    """
    def __init__(self, path, inst_id, inst_type='spot', freq='2.5Hz', readout='bulk'):
        """
        Initialize a SPOT object by reading data from the specified directory, setting up
        relevant attributes, and optionally loading raw and bulk data.

        Parameters:
            path (str): The path to the spotter data directory.
            inst_id (str): The instrument's ID.
            inst_type (str): The instrument type ('spot' by default).
            freq (str): The frequency of the Spotter displacement recordings (default is 2.5Hz).
            readout (str): Specifies which data to read ('bulk' by default; options are 'bulk', 'raw', or 'all').

        Initializes:
            self.dpl_raw: Raw displacement data.
            self.dpl: Processed displacement data.
            self.spectra: Computed wave spectra.
            self.ts: Start time of instrument activity.
            self.te: End time of instrument activity.
        """
        print('Initializing %s' % inst_id)
        super().__init__(inst_id, inst_type, freq)
        self.dir = path  # Spotter directory
        self.rdir = join(path, 'full/raw')  # Raw data directory
        self.bdir = join(path, 'bulk')  # Bulk data directory

        # Initialize attributes which will be defined in other methods
        self.dpl_raw = None  # Attribute that will hold raw displacement data
        self.dpl = None  # Attribute that will hold processed displacement data
        self.spectra = None  # Attribute that will hold computed wave spectra
        self.ts = None  # Start time instrument activity
        self.te = None  # End time instrument activity

        # read raw data
        if readout.lower() in ['raw', 'full', 'all ']:
            print('Reading raw data...')
            self.vars = {
                'dpl': ['displacement', ['x', 'y', 'z']],
                'sst': ['sst', ['sst']],
                'loc': ['location', ['lat', 'lon']],
                'spc': ['Szz', ['dof'] + list(np.linspace(0, 1.240234375, 128))]
            }

            date_cols = ['year', 'month', 'day', 'hour', 'minute', 'second', 'millisecond']

            for var in self.vars:
                print('--- %s.csv ...' % self.vars[var][0])
                df = (pd.read_csv(join(self.rdir, '%s.csv' % (self.vars[var][0])))
                      .set_axis(date_cols + self.vars[var][1], axis=1))
                df['datetime'] = pd.to_datetime(df[date_cols])
                df = df.set_index('datetime', drop=True).drop(date_cols, axis=1)
                setattr(self, '%s_raw' % var, df.copy())  # store raw data
                setattr(self, var, df.copy())  # store another copy for processing

        else:
            self.vars = {}

        # read bulk data
        print('Reading bulk data...')
        self.bulk_vars = {
            'Hs': ['m', 'wave'],
            'Tp': ['s', 'wave'],
            'Tm': ['s', 'wave'],
            'pdir': [r'$\degree$', 'wave'],
            'pdir_spread': [r'$\degree$', 'wave'],
            'mdir': [r'$\degree$', 'wave'],
            'mdir_spread': [r'$\degree$', 'wave'],
            'U': ['m/s', 'wind'],
            'Udir': [r'$\degree$', 'wind'],
            'SST': [r'$\degree$C', 'climate'],
            'RH': ['%', 'climate'],
        }
        cols0 = ['batV', 'P', 'RH', 'datetime', 'Hs', 'Tp', 'Tm', 'pdir', 'pdir_spread', 'mdir', 'mdir_spread',
                 'lat', 'lon', 'U', 'Udir', 'SST', 'psource']

        # read data
        self.bulk_df = (pd.read_csv(join(self.bdir, '%s_bulk_raw.csv' % self.id), na_values=['-'])
                        .dropna(axis=1, how='all')
                        .set_axis(cols0, axis=1)
                        .set_index('datetime')
                        .drop(['batV', 'P', 'psource'], axis=1)
                        .sort_index()
                        )
        self.bulk_df.index = pd.to_datetime(self.bulk_df.index, unit='s')

    def __getstate__(self):
        """
        Return the state of the SPOT object for pickling, excluding displacement data to reduce size.

        Returns:
            dict: A dictionary containing the state of the object excluding the displacement data.
        """
        d = self.__dict__
        self_dict = {k: d[k] for k in d if 'dpl' not in k}
        return self_dict

    def read_df(self, variables='dpl', mod='both', ext='xz'):
        """
        Load data into the SPOT object. Data should be in the form of a pickled pandas DF, inside
        the working directory and named as 'INSTID_VAR_mod.ext', where INSTID is the unique instrument identifier,
        VAR is the variable, and mod is raw or processed data.

        Parameters:
        - variables (list of str): Which variables to load, default is ony displacement data ('dpl')
        - mod (str): Which data to load ('raw', 'proc', or 'both')
        - ext (str): Extension of the file containing the data.
        """
        if not isinstance(variables, list):
            variables = [variables]

        for var in variables:
            if mod in ['raw', 'both']:
                self.dpl_raw = pd.read_pickle(join(self.dir, '%s_%s_raw.%s' % (self.id, var, ext)))
            if mod in ['proc', 'both']:
                self.dpl = pd.read_pickle(join(self.dir, '%s_%s_proc.%s' % (self.id, var, ext)))

    ##### DATA PROCESSING
    def time_offset(self, hours=0, minutes=0, seconds=0, df_keys=None):
        """
        Adjust the timestamps of specified DataFrames by adding a specified time offset.

        This method adds a `timedelta` to the timestamps of the selected DataFrames. For subtracting time,
        negative values can be passed for `hours`, `minutes`, or `seconds`.

        Parameters:
            hours (int, optional): Number of hours to add to the timeseries. Default is 0.
            minutes (int, optional): Number of minutes to add to the timeseries. Default is 0.
            seconds (int, optional): Number of seconds to add to the timeseries. Default is 0.
            df_keys (list of str, optional): List of DataFrame keys (e.g., 'raw', 'proc') to apply the offset to.
                                             If None, defaults to all keys.

        Raises:
            AttributeError: If a specified DataFrame key is not found in the object.
        """
        if df_keys is None:
            df_keys = [key for key in self.__dict__.keys() if key in self.vars] + ['bulk_df']
        elif not isinstance(df_keys, list):
            df_keys = [df_keys]

        # Create a timedelta object from the provided hours, minutes, and seconds
        time_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)

        # Apply the time offset to each specified DataFrame
        for key in df_keys:
            try:
                # Adjust the index of the DataFrame by the computed timedelta
                self.__getattribute__('%s' % key).index += time_delta
            except AttributeError:
                raise AttributeError(f"DataFrame with key '{key}' not found.")

    def active_dates(self, ts, te, variables=None):
        """
        Filter the data to a specific time range for analysis.

        Parameters:
            ts (datetime-like): Start time of the active period.
            te (datetime-like): End time of the active period.
            variables (list of str, optional): Variables to apply the filtering to. Defaults to all.

        Note:
        This method directly modifies the attributes corresponding to the variable names in `vars` to contain only data
        within the specified time range (`ts` to `te`). The modification is done in place.
        """
        self.ts = ts
        self.te = te

        if variables is None:
            variables = list(self.vars.keys())
        elif not isinstance(variables, list):
            variables = [variables]

        for var in variables:
            self.__dict__[var] = self.__dict__[var].loc[(self.__dict__[var].index > ts) & (self.__dict__[var].index < te)]

    def skip_dates(self, times, variables=None, interpolate=False):
        """
        Exclude data within specified time ranges, and optionally interpolate the gaps.

        Parameters:
            times (list of tuples): List of (start, end) time ranges to exclude.
            variables (list of str, optional): Variables to apply the exclusion to. Defaults to all.
            interpolate (bool, optional): Whether to interpolate over excluded periods. Default is False.

        Note:
        This method directly modifies the attributes corresponding to the variable names in `vars` by setting the data 
        to NaN for the specified `times` periods. If `interpolate` is True, the gaps introduced by setting values to NaN 
        are filled using linear interpolation.
        """
        if variables is None:
            variables = list(self.vars.keys())
        elif not isinstance(variables, list):
            variables = [variables]

        for var in variables:
            for t in times:
                self.__dict__[var].loc[(self.__dict__[var].index > t[0]) &
                                       (self.__dict__[var].index < t[1]), :] = np.nan

            if interpolate:
                self.__dict__[var] = self.__dict__[var].interpolate()

    def compute_spectra(self, mod='ts', t_range=None, window=30, interval=30, fs=2.5, nBlocks=16, plot=False, xax='f'):
        """
        Compute wave spectra from the displacement timeseries for a given time range using FFT, and optionally plot.

        Parameters:
        mod : str, optional
            Mode of computation. Can be 'single' for a single spectrum calculation or
            'ts', 'timeseries', or 'rolling' for a rolling window approach.
            The default is 'ts'.
        t_range : list of datetime-like, optional
            The start and end times for the data range to compute the spectrum.
            If None, uses the entire dataset range. The default is None.
        window : int, optional
            The window size in minutes for the rolling computation or single spectrum length.
            The default is 30.
        interval : int, optional
            The interval in minutes between rolling windows for the 'rolling' mode.
            The default is 30.
        fs : float, optional
            The sampling frequency in Hz. The default is 2.5.
        nBlocks : int, optional
            The number of blocks to divide the data into for FFT. The default is 16.
        plot : bool, optional
            If True, plots the computed spectrum. The default is False.
        xax : str, optional
            Determines the x-axis of the plot. 'f' for frequency in Hz, 'T' for period in seconds.
            The default is 'f'.
        """
        # If no time frame is specified, use the entire timeseries
        if t_range is None:
            t_range = [self.ts, self.te]

        data = self.dpl.loc[t_range[0]:t_range[1]].z
        N = len(data)

        # Single spectrum
        if mod.lower() == 'single':
            t_ref = t_range[0] + (t_range[1] - t_range[0]) / 2
            E, f, Hs, Tp, Tm01, Tm02 = vardens_spectrum(data.dropna(), N / nBlocks, fs).values

            # plot single spectrum
            if plot:
                _, axs = plt.subplots(1, 1, figsize=[8, 4], sharex=True)

                if xax == 'f':
                    axs.plot(f, E)
                    axs.fill_between(f, 0, E, alpha=0.2)
                    axs.set(xlim=[0, 1], ylim=[0, None],
                            xlabel='Wave frequency [Hz]', ylabel='Energy density [m2/Hz]',
                            title='%s wave spectrum at %s (%d blocks), Hs = %.02f m, Tp = %.01f s' % (
                                self.id, t_ref.strftime('%d/%m %H:%M'), nBlocks, Hs, Tp))
                elif xax == 'T':
                    axs.plot(1 / f, E)
                    axs.fill_between(1 / f, 0, E, alpha=0.2)
                    axs.set(xlim=[1, 40], ylim=[0, None],
                            xlabel='Wave period [s]', ylabel='Energy density [m2/Hz]',
                            title='%s wave spectrum at %s (%d blocks), Hs = %.02f m, Tp = %.01f s' % (
                                self.id, t_ref.strftime('%d/%m %H:%M'), nBlocks, Hs, Tp))
                axs.grid()

        # Multiple spectra over time series
        elif mod.lower() in ['ts', 'timeseries', 'rolling']:
            print('Computing wave spectra...')
            window_n = window * 60 * fs + 1  # window length in data points
            interval_n = interval * 60 * fs  # interval length in data points
            n_windows = int((N - window_n) / interval_n + 1)  # number of windows
            df = fs / int(window_n / nBlocks - ((window_n / nBlocks) % 2))  # frequency resolution
            f = np.arange(0, fs / 2 + df, df)  # frequency axis
            self.spectra = pd.DataFrame(columns=['Hs', 'Tp', 'Tm01', 'Tm02'] + list(f))  # Empty DF to store results

            # Process each window
            for i in tqdm(range(n_windows)):
                t_start = self.dpl.loc[t_range[0]:t_range[1]].index[int(i * interval_n)]  # start time current window
                t_stop = t_start + timedelta(minutes=window)  # end time current window
                t_ref = t_start + (t_stop - t_start) / 2  # reference time current window
                temp = data.loc[t_start:t_stop].copy()  # data current window

                # Compute spectral parameters and store in df
                spec = vardens_spectrum(temp.dropna(), window_n / nBlocks, fs)
                self.spectra.loc[t_ref] = [spec['Hs'], spec['Tp'], spec['Tm01'], spec['Tm02']] + list(spec['vardens'])

    ##### PLOTTING
    def plot_ts(self, var, t_range=None, ax=None, resample=None, hlines=[], vlines=[], outpath=None, show=1, **kwargs):
        """
        Plot time series data for a specified variable within a time range.

        Parameters:
            var (str): The variable to plot.
            t_range (list/tuple, optional): The start & end times of the plot. Default is full range.
            ax (matplotlib axis, optional): The axis to plot on. Default is None.
            resample (str, optional): Resampling frequency for the plot. Default is None.
            hlines (list, optional): List of y-values for horizontal reference lines. Default is None.
            vlines (list, optional): List of x-values (timestamps) for vertical reference lines. Default is None.
            outpath (str, optional): Path to save the figure. Default is None.
            show (bool, optional): Whether to display the plot. Default is True.
            **kwargs: Additional keyword arguments for plotting (e.g., linewidth, marker).
        """
        if isinstance(t_range, (list, tuple)):
            ts, te = t_range
        else:
            ts, te = self.ts, self.te

        if resample is None:
            plot_df = self.__dict__['%s' % var].loc[ts:te]
        else:
            plot_df = self.__dict__['%s' % var].loc[ts:te].resample(resample).mean()

        N = len(plot_df.columns)

        if not ax:
            fig, axs = plt.subplots(N, 1, figsize=(10, 2 * N), sharex=True)
            axs = [axs] if N == 1 else axs

            for i, col in enumerate(plot_df.columns):
                plot_df[col].plot(ax=axs[i], label=col, c='C%d' % i, **kwargs)
                axs[i].set(ylabel=col, xlim=(ts, te))

            for ax in axs:
                if hlines:
                    ax.hlines(hlines, ts, te, ls=':', color='C3')
                if vlines:
                    ax.vlines(vlines, ts, te, ls=':', color='C3')
                ax.grid()

            plt.tight_layout()

        if show:
            plt.show()

        if outpath:
            plt.savefig(outpath, dpi=350, bbox_inches='tight')

    ##### SAVING
    def save_obj(self, ext='p'):
        """
        Save the SPOT object (excluding DataFrames) to a file using pickle or lzma.

        Parameters:
            ext (str): The file extension ('p' for pickle, 'xz' for lzma-compressed pickle).
        """
        fname = '%s.%s' % (self.id, ext)

        if ext == 'p':
            with open(join(self.dir, fname), 'wb') as f:
                pickle.dump(self, f)
        else:
            with lzma.open(join(self.dir, fname), 'wb') as f:
                pickle.dump(self, f)

    def save_df(self, variables='dpl', mod='proc', ext='parq'):
        """
        Save dataframes to a pickle file
        
        Parameters:
        - variables (list of str): Variables for which df should be saved (default only 'dpl')
        - mod : {'raw', 'proc', 'both'}, default 'proc'
            Specifies whether to save the raw or processed data
        - ext : {'parq', 'xz', 'gz', 'p'}, default 'parq'
            The file extension to use for the output file.
            - 'parq': Save as Apache parquet file
            - 'xz': Save as .xz compressed pickle file
            - 'gz': Save as .gz (gzip) compressed pickle file
            - 'p': Save to uncompressed pickle file
        """
        if not isinstance(variables, list):
            variables = [variables]

        for var in variables:
            print('Saving %s data to .%s file...' % (var, ext))

            if mod.lower() in ['raw', 'both']:
                if ext not in ['parq', 'parquet']:
                    self.__dict__['%s_raw' % var].to_pickle(join(self.dir, '%s_%s_raw.%s' % (self.id, var, ext)))
                else:
                    self.__dict__['%s_raw' % var].to_parquet(join(self.dir, '%s_%s_raw.%s' % (self.id, var, ext)))
            if mod.lower() in ['proc', 'both']:
                if ext not in ['parq', 'parquet']:
                    self.__dict__['%s' % var].to_pickle(join(self.dir, '%s_%s_proc.%s' % (self.id, var, ext)))
                else:
                    self.__dict__['%s' % var].to_parquet(join(self.dir, '%s_%s_proc.%s' % (self.id, var, ext)))


# Class to store and process cross-shore GPS profiles
class CrossShoreProfiles:
    """
    A class for managing and processing cross-shore GPS profiles.

    The CrossShoreProfiles class is designed to transform and manage cross-shore profiles
    by applying a linear fit to create a new coordinate system. It supports the addition of
    instrument data, profile plotting, and saving/loading profile data for further analysis.

    Attributes:
        id (str): Identifier for the profile set.
        base_id (str): Identifier for the base profile used to define the new coordinate system.
        base_date (datetime): Date of the base profile.
        linfit (ndarray): Linear fit coefficients (slope and intercept) for the base profile.
        angle (float): Angle of the new x-axis relative to true north.
        slope (float): Slope of the new x-axis in the transformed coordinate system.
        new_orig (tuple): Origin of the new coordinate system.
        new_orig_wsg (tuple): WGS84 coordinates of the origin.
        profiles (dict): Dictionary containing transformed profiles with new x, y coordinates.
        instruments (pd.DataFrame or None): DataFrame containing instrument locations (optional).

    Methods:
        add_instruments: Add instrument positions to the profile system and transform their coordinates.
        plot_profile: Plot a single profile in the transformed coordinate system.
        plot_profiles: Plot multiple profiles in the transformed coordinate system.
        save_obj: Save the CrossShoreProfiles object to a file using pickle.
    """

    def __init__(self, prof_id, base_id, profiles, xname='e', yname='n', new_orig='first', crs0='EPSG:6587'):
        """
        Initialize the CrossShoreProfiles object and transform profiles into a new coordinate system.

        Parameters:
            prof_id (str): Identifier for the profile set.
            base_id (str): Identifier for the base profile used to set the new coordinate system.
            profiles (dict): Dictionary containing the profiles to be transformed.
            xname (str, optional): Name of the column representing the x-coordinates (default is 'e' for easting).
            yname (str, optional): Name of the column representing the y-coordinates (default is 'n' for northing).
            new_orig (str, optional): Whether to use the 'first' or 'last' point of the base profile as the origin
                                      for the new coordinate system (default is 'first').
            crs0 (str, optional): Coordinate reference system (CRS) for the original profiles (default is 'EPSG:6587').

        Notes:
            This method calculates a linear fit for the base profile and transforms each profile's coordinates
            into a new system based on this fit. The new x-axis is aligned with the base profile, and the origin
            is projected to the line defined by the fit.
        """
        self.id = prof_id
        self.base_id = base_id
        self.base_date = datetime(2023, int(base_id[:2]), int(base_id[2:]))

        base = profiles[base_id]

        if new_orig == 'last':
            i_orig = -1
        else:
            i_orig = 0

        # Create linear fit to base profile, which will be the new x-axis
        self.linfit = np.polyfit(base[xname], base[yname], 1)
        self.angle = np.degrees(np.pi / 2 - np.arctan(self.linfit[0]))  # Angle x-axis relative to positive y-axis (N)
        self.slope = np.tan(np.radians(90 - self.angle))  # slope of the new x-axis
        self.new_orig = project_point_to_line((base[xname].iloc[i_orig], base[yname].iloc[i_orig]), *self.linfit)
        self.new_orig_wsg = Transformer.from_crs(crs0, 'EPSG:4326').transform(*self.new_orig)

        # convert each profile to new coordinate system and add new coordinates as xy-columns
        for prof in profiles:
            profiles[prof]['x'], profiles[prof]['y'] = (zip(*profiles[prof].apply(lambda row: transform_point(
                (row[xname], row[yname]), self.linfit[0], self.linfit[1], self.new_orig), axis=1)))
            profiles[prof] = profiles[prof].sort_values('x')
            profiles[prof].index = ['P%02d' % i for i in range(1, len(profiles[prof]) + 1)]

        self.profiles = profiles.copy()

        # Empty attribute for instruments, can be added through add_instruments
        self.instruments = None

    def __str__(self):
        return 'Cross-shore profiles for %s, transect angle = %.1f degrees' % (self.id, self.angle)

    def __repr__(self):
        return self.__str__()

    def add_instruments(self, instruments, xname, yname):
        """
        Add instrument positions to the profile system and transform their coordinates.

        Parameters:
            instruments (pd.DataFrame): DataFrame containing instrument positions.
            xname (str): Column name for x-coordinates in the instrument data.
            yname (str): Column name for y-coordinates in the instrument data.

        Notes:
            This method applies the same transformation to the instrument data as was done for the profiles,
            converting instrument coordinates to the new coordinate system.
        """
        instruments['x'], instruments['y'] = (zip(*instruments.apply(lambda row: transform_point(
            (row[xname], row[yname]), self.linfit[0], self.linfit[1], self.new_orig), axis=1)))

        self.instruments = instruments.copy()

    ### PLOTTING
    def plot_profile(self, prof, color=None, title=None, outpath=None, show=1):
        """
        Plot a single profile in the new coordinate system.

        Parameters:
            prof (str): Identifier of the profile to plot.
            color (str, optional): Color for the plot (default is 'C0').
            title (str, optional): Title of the plot (default is None).
            outpath (str, optional): Path to save the plot as an image (default is None).
            show (int, optional): Flag to control whether to display the plot (1 to display, 0 to skip). Default is 1.

        Notes:
            This method generates three subplots:
            - Cross-shore distance vs. elevation.
            - Original easting and northing coordinates.
            - Projection distance in the transformed coordinate system.
        """
        if color is None:
            color = 'C0'

        fig, axs = plt.subplots(1, 3, figsize=(12, 3))

        axs[0].plot(self.profiles[prof].x, self.profiles[prof].z, c=color)
        axs[0].set(xlim=(self.profiles[prof].x.min(), self.profiles[prof].x.max()),
                   xlabel='Cross-shore distance [m]', ylabel='Elevation [m NAVD88]')
        axs[0].grid()

        axs[1].axline((0, self.linfit[1]), slope=self.linfit[0], c='k', lw=0.5,
                      label='Transect axis (%s)' % self.base_date.strftime('%d-%b'))
        axs[1].scatter(self.profiles[prof].e, self.profiles[prof].n, color=color, s=10, label='GPS points')
        axs[1].set(xlim=(self.profiles[prof].e.min() - 3, self.profiles[prof].e.max() + 3),
                   ylim=(self.profiles[prof].n.min() - 3, self.profiles[prof].n.max() + 3),
                   xlabel='Easting [m]', ylabel='Northing [m]')
        axs[1].legend()

        axs[2].scatter(self.profiles[prof].x, np.abs(self.profiles[prof].y), color=color, s=5)
        axs[2].vlines(self.profiles[prof].x, 0, np.abs(self.profiles[prof].y), color=color)
        axs[2].set(xlim=(self.profiles[prof].x.min(), self.profiles[prof].x.max()), ylim=(0, None),
                   xlabel='Cross-shore distance (x) [m]', ylabel='Projection distance (y) [m]')

        if title:
            fig.suptitle(title)

        fig.tight_layout()

        if outpath:
            fig.savefig(outpath, dpi=350)

        if show:
            plt.show()

    def plot_profiles(self, profs=None, ax=None, title=None, outpath=None, show=1, **kwargs):
        """
        Plot multiple profiles in the new coordinate system.

        Parameters:
            profs (list of str, optional): List of profile identifiers to plot. If None, plots all profiles.
            ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new plot is created.
            title (str, optional): Title for the plot (default is None).
            outpath (str, optional): Path to save the plot as an image (default is None).
            show (int, optional): Flag to control whether to display the plot (1 to display, 0 to skip). Default is 1.
            **kwargs: Additional keyword arguments for the plotting function.

        Notes:
            This method plots each profile's cross-shore distance vs. elevation, using a different color
            for each profile. If no profile list is provided, all profiles are plotted.
        """
        if not ax:
            fig, ax = plt.subplots()

        if not profs:
            profs = self.profiles

        for prof in profs:
            prof_date = datetime(2023, int(prof[:2]), int(prof[2:])).strftime('%d-%b')
            ax.plot(profs[prof].x, profs[prof].z, label=prof_date, **kwargs)

        ax.grid()
        ax.legend()
        ax.set(xlim=(0, None), xlabel='Cross-shore distance [m]', ylabel='Elevation [m NAVD88]', title=title)
        fig.tight_layout()

        if outpath:
            fig.savefig(outpath, dpi=350)

        if show:
            plt.show()

    def save_obj(self, outpath):
        """
        Save the CrossShoreProfiles object to a file using pickle.

        Parameters:
            outpath (str): Path where the object should be saved.

        Notes:
            This method serializes the current state of the CrossShoreProfiles object, allowing
            it to be reloaded later. It does not save DataFrames or any dynamic data.
        """
        with open(outpath, 'wb') as f:
            pickle.dump(self, f)


### Standalone functions
def read_inst(fdir, fname=None, read_data=None, data_ext='parq'):
    """
    Read an instrument class object from a pickle file and load its data if specified.

    Parameters:
        fdir (str): Directory where the pickle file is located.
        fname (str, optional): Name of the class object file. If not provided, it is inferred from the directory name.
        read_data (str, optional): Which data to load ('raw', 'proc', 'both', or None). Default is None.
        data_ext (str, optional): Extension of the data files to load (default is 'parq').

    Returns:
        inst (Instrument or None): The loaded instrument object, or None if loading failed.

    Raises:
        FileNotFoundError: If the specified file does not exist in the directory.
        ValueError: If the file extension is unsupported.
        pickle.PickleError: If there is an error during unpickling.
    """
    if not fname:
        fname = fdir.split('/')[-1]
        fdir = fdir[:-len(fname)]

    ext = fname.split('.')[-1]

    try:
        if ext == 'p':
            with open(join(fdir, fname), 'rb') as f:
                inst = pickle.load(f)
        elif ext == 'xz':
            with lzma.open(join(fdir, fname), 'rb') as f:
                inst = pickle.load(f)
        else:
            raise ValueError("Unsupported file extension")
    except FileNotFoundError:
        print(f"Error: File '{fname}' not found in directory '{fdir}'")
        return None
    except pickle.PickleError as e:
        print(f"Error: Failed to unpickle object from file '{fname}': {e}")
        return None

    # update working directory
    inst.dir = '/'.join(join(fdir, fname).split('/')[:-1])

    if read_data:
        inst.read_df(dfs=read_data, ext=data_ext)

    return inst


def nth_spectral_moment(frequencies, spectral_values, n):
    """
    Calculate the n'th moment of a variance density spectrum.

    Parameters:
        frequencies (array_like): Array of frequencies.
        spectral_values (array_like): Array of spectral values corresponding to the frequencies.
        n (int): The order of the moment to compute.

    Returns:
        float: The n'th moment of the spectrum.
    """
    delta_f = frequencies[1] - frequencies[0]  # Assuming equidistant frequencies
    moment = np.sum(spectral_values * (frequencies ** n) * delta_f)
    return moment


def vardens_spectrum(data, Fs, nfft=None):
    """
    Compute the variance density spectrum of a time series and derive spectral wave parameters.

    Parameters:
        data (array_like): Time series data of surface elevation (e.g., from wave measurements).
        Fs (float): Sampling frequency of the data in Hz.
        nfft (int, optional): Length of the data blocks for the Fourier transform (in data points). If not provided,
                              the entire time series is treated as one block.

    Returns:
        dict: A dictionary containing:
            - 'freq' (array): Discrete frequencies at which the variance density spectrum is computed.
            - 'vardens' (array): Variance density at each frequency.
            - 'Hs' (float): Significant wave height.
            - 'Tp' (float): Peak wave period.
            - 'Tm01' (float): Mean wave period (first moment).
            - 'Tm02' (float): Mean wave period (second moment).

    Notes:
        This function divides the data into blocks, detrends each block, and applies a Fourier
        transform to compute the variance density spectrum. From this spectrum, wave parameters
        such as significant wave height and peak period are derived.
    """

    # PROCESSING
    n = len(data)
    if not nfft:
        nfft = len(data)    # if no nfft is given, the time series is one big block
    else:
        nfft = int(nfft - (nfft % 2))   # make block length even
    nBlocks = int(n / nfft)  # number of blocks
    data = data[0:nBlocks * nfft]  # only take complete blocks
    dataBlock = np.reshape(data, (nBlocks, nfft))  # each row is one block

    # detrend each datablock so that the mean surface elevation is 0
    dataBlock.setflags(write=True)  # make datablock array changable/writable to detrend it
    for i in range(nBlocks):
        dataBlock[i, :] = detrend(dataBlock[i, :])

    # CALCULATION VARIANCE DENSITY SPECTRUM
    df = Fs / nfft  # frequency resolution of the spectrum df = 1/[Duration of one block]
    f = np.arange(0, Fs / 2 + df, df)  # frequency axis (Fs/2 = Fnyquist = max frequency)
    fId = np.arange(0, len(f))

    # Calculate the variance for each block and for each frequency
    fft_data = fft(dataBlock, n=nfft, axis=1)  # Fourier transform of the data
    fft_data = fft_data[:, fId]         # Only one side needed
    A = 2.0 / nfft * np.real(fft_data)  # A(i,b) and B(i,b) contain the Fourier coefficients Ai and Bi
    B = 2.0 / nfft * np.imag(fft_data)
    E = (A ** 2 + B ** 2) / 2           # E(i,b) = ai^2/2 = variance at frequency fi for block b.

    # Average the variance over the blocks and divide by df to get the variance density spectrum
    E = np.mean(E, axis=0) / df

    # Compute spectral wave parameters
    Hs = 4 * np.sqrt(nth_spectral_moment(f, E, 0))   # Significant wave height
    if f[np.argmax(E)] != 0:                            # Peak period + inverse of peak frequency
        Tp = 1 / f[np.argmax(E)]                        # If peak freq = 0 then Tp = NaN
    else:
        Tp = np.nan
    Tm01 = (nth_spectral_moment(f, E, 1) / nth_spectral_moment(f, E, 0)) ** -1      # Mean period m01
    Tm02 = np.sqrt(nth_spectral_moment(f, E, 0) / nth_spectral_moment(f, E, 2))     # Mean period m02

    spectrum = {
        'freq': f,
        'vardens': E,
        'Hs': Hs,
        'Tp': Tp,
        'Tm01': Tm01,
        'Tm02': Tm02,
    }

    return spectrum


def project_point_to_line(point, slope, intercept):
    """
    Project a point onto a straight line defined by its slope and intercept.

    Parameters:
        point (tuple of float): Coordinates of the point to be projected in the form (x, y).
        slope (float): Slope of the line onto which the point is projected.
        intercept (float): Y-intercept of the line onto which the point is projected.

    Returns:
        tuple of float: The projected point on the line in the form (x, y).

    Notes:
        The projection is computed geometrically, minimizing the perpendicular distance between
        the point and the line.
    """
    x1, y1 = point
    x2 = (x1 + slope * (y1 - intercept)) / (slope ** 2 + 1)
    y2 = slope * x2 + intercept
    return x2, y2


def transform_point(point, slope, intercept, new_orig=None):
    """
    Transform a point into a local coordinate system defined by a new x-axis.

    Parameters:
        point (tuple of float): Coordinates of the point in the original coordinate system.
        slope (float): Slope of the line defining the new x-axis.
        intercept (float): Y-intercept of the line defining the new x-axis.
        new_orig (tuple of float, optional): Origin of the new coordinate system. If None, the
                                             origin is set to (0, intercept) on the new x-axis.

    Returns:
        tuple of float: The transformed coordinates (x, y) in the new coordinate system.

    Notes:
        This function projects the point onto the new x-axis and computes its new coordinates
        relative to the specified origin. The x-coordinate is the distance along the line,
        and the y-coordinate is the perpendicular distance from the line.
    """
    if new_orig is None:
        new_orig = (0, intercept)

    # Projection of the point onto the line
    x_proj = (point[0] + slope * (point[1] - intercept)) / (1 + slope ** 2)
    y_proj = slope * x_proj + intercept

    # x coordinate is the distance along the line from the origin
    x = np.sqrt((x_proj - new_orig[0]) ** 2 + (y_proj - new_orig[1]) ** 2)
    if x_proj < new_orig[0]:
        x = -x

    # y coordinate is the perpendicular distance to the line
    y = np.sqrt((point[0] - x_proj) ** 2 + (point[1] - y_proj) ** 2)
    if y_proj > point[1]:
        y = -y

    return x, y


