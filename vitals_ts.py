"""
This package provides vitals timeseries processing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class VitalsTS():
    """
    This class cleans and interpolates vitals timeseries.
    
    Args:
        raw (df): Timeseries data, columns as follows:
            SubjectID (number): unique subject ID for each subject
            Timestamp (datetime): timestamp of measurement recording
            One column for each measure to be analyzed (e.g. MAP, SBP, DBP), all type number
        static (df): Static data about subjects, minimum columns as follows:
            SubjectID (number): unique subject ID for each subject
            Start (datetime): Timestamp of event start
            Stop (datetime): Timestamp of event stop
            
    Properties:
        self.raw (df): Raw timeseries data
        self.static (df): Static data about subjects
        self.processed (df): Filtered timeseries data before imputation
        self.interpolated (df): Imputed timeseries data
        self.qa (df): Quality metrics for imputed data, one subject per row

    Methods:
        Filters
            filter_multiple_at_time
            filter_na_rows
            filter_cross_column
            filter_absolute_range
            filter_lab_spikes
        Interpolation
            interpolate
    """
    
    def __init__(self, raw, static):
        """
        
        """
        self.raw = raw
        self.static=static
        self.processed = raw

        
    def filter_multiple_at_time(self):
        """
        If subject has multiple measurements at same time, randomly keeps one
        """
        self.processed = self.processed.groupby(['SubjectID','Timestamp']).agg(pd.DataFrame.sample).reset_index()
        
        
    def filter_na_rows(self,how='all'):
        """
        Remove rows with na
        
        Args:
            how (str): 'all': remove row if all entries are na, 'any': remove row if any are na (not recommended)
        """
        self.processed = self.processed.dropna(how=how)
        
        
    def filter_cross_column(self, func):
        """
        Enforces dependencies between measure columns
        
        Args:
            func (method): A method that takes a df as first argument and returns a (filtered) df, e.g.:
                # Drop if DBP > MAP
                def filter_bps(x):
                    return x[~(x['DBP'] > x['SBP'])]
                v.filter_cross_column(filter_bps)
        """
        
        self.processed = func(self.processed)
        
        
    def filter_absolute_range(self, col, low, high):
        """
        Removes values outside closed interval
        
        Args:
            col (str): Name of column to filter
            low (number): Values less than this will be replaced with na
            high (number): Values greater than this will be replaced with na
        """
        self.processed.loc[(self.processed[col] < low) | (self.processed[col] >high), col] = np.NaN
        
        
    def filter_lab_spikes(self, col, window=3, thresh=180, time_unit='m', plot=False):
        """
        cols: which columns to filter
        """
        
        subject_ids = self.processed['SubjectID'].drop_duplicates()
        filter_flags = pd.DataFrame()
        
        for s in subject_ids:
            # Get data for subject
            dat = self.processed[self.processed['SubjectID'] == s].copy()
            
            # Connnvert timestamp to number, 0 = start
            dat['Timestamp'] = (dat['Timestamp'] - dat['Timestamp'].min()) / np.timedelta64(1,time_unit)
            dat.sort_values('Timestamp', inplace=True)
            
            # Drop NA
            dat_sub = dat[['Timestamp',col]].copy()
            dat_sub = dat_sub.dropna(how='any')
            
            # Get 1st and 2nd diff
            dat_sub['Slope'] = dat_sub[col].diff(1)/dat_sub['Timestamp'].diff(1)
            dat_sub['Slope_diff'] = dat_sub['Slope'].diff(-1)
            
            # Flag as lab spike if 2nd diff > threshold
            dat['{}_lab_spike'.format(col)] = dat_sub['Slope_diff'] > thresh
            dat['{}_lab_spike'.format(col)].fillna(False, inplace=True)

            # Add flags to self.processed
            lab_spike_idx = list(dat[dat['{}_lab_spike'.format(col)]].index)
            self.processed.loc[lab_spike_idx, col] = np.NaN
            self.processed.loc[lab_spike_idx, '{}_lab_spike'.format(col)] = True
   
            # Plot waveform
            if plot:
                fig, ax = plt.subplots(figsize=(40,3))
                for c in cols:
                    plt.plot(dat['Timestamp'], dat[col])
                    for i, r in dat[dat['{}_lab_spike'.format(col)]].iterrows():
                        plt.scatter(r['Timestamp'], r[col], color='red')
                    
                plt.show()
       
        # Format self.processed col as int
        self.processed[['{}_lab_spike'.format(col)]] = self.processed[['{}_lab_spike'.format(col)]].fillna(False).astype(int)

        
    def interpolate(self, col, point_valid_time=5, step_size=1, time_unit='m', how='linear'):
        # Get subject IDs
        subject_ids = self.processed['SubjectID'].drop_duplicates()
        
        # Containers
        self.interpolated = pd.DataFrame()
        self.qa = pd.DataFrame()
        
        for s in subject_ids:
            # Get data for subject
            dat = self.processed[self.processed['SubjectID'] == s].copy()
            
            # Get only those within start/stop
            start = self.static[self.static['SubjectID']==s].iloc[0]['Start']
            stop = self.static[self.static['SubjectID']==s].iloc[0]['Stop']
            dat = dat[(dat['Timestamp'] >= start) & (dat['Timestamp'] <= stop)]
            
            # Ensure sorted by time
            dat.sort_values('Timestamp', inplace=True)
            
            if dat.shape[0] > 0:
                # Adjust time to start
                dat['Timestamp'] = (dat['Timestamp'] - start)/np.timedelta64(1,'m')
                
                # Get sparsity metrics
                qa_sub = {'SubjectID':s}
                qa_sub['duration'] = (stop - start)/np.timedelta64(1,'m')
                qa_sub['time_start'] = dat['Timestamp'].min()
                qa_sub['time_stop'] = dat['Timestamp'].max()
                qa_sub['time_diff_max'] = dat['Timestamp'].diff(1).max()
                
                # Set interpolated flag
                dat['{}_interpolated'.format(col)] = False
                    
                # Fill time gaps
                ts = pd.DataFrame().from_dict({'Timestamp':np.arange(qa_sub['duration'],step=step_size)})
                dat = pd.merge(ts,dat, on='Timestamp',how='left')
                dat['SubjectID'] = s
                
                # Get missingness pre-interp
                qa_sub['{}_min_missing_total'.format(col)] = dat[col].isna().sum()*step_size
                qa_sub['{}_min_missing_between'.format(col)] = dat.iloc[int(qa_sub['time_start']):int(qa_sub['time_stop'])+1][col].isna().sum()*step_size
                qa_sub['{}_min_missing_ends'.format(col)] = qa_sub['{}_min_missing_total'.format(col)] - qa_sub['{}_min_missing_between'.format(col)]
                    
                 # Interpolate
                dat[col] = dat[col].interpolate(method=how, limit=point_valid_time, limit_area='inside')

                # Get missingness post-interp
                qa_sub['{}_min_missing_total_after_interp'.format(col)] = dat[col].isna().sum()*step_size
                qa_sub['{}_min_missing_between_after_interp'.format(col)] = dat.iloc[int(qa_sub['time_start']):int(qa_sub['time_stop']+1)][col].isna().sum()*step_size
                dat['{}_interpolated'.format(col)] = dat.apply(lambda x: True if (~np.isnan(x[col])) & (np.isnan(x['{}_interpolated'.format(col)])) else x['{}_interpolated'.format(col)] if not x['{}_interpolated'.format(col)] else np.NaN, axis=1)

                # Append to containers
                self.interpolated = pd.concat([self.interpolated,dat], ignore_index=True)
                self.qa = self.qa.append(qa_sub, ignore_index=True)
                self.qa = self.qa.groupby('SubjectID').agg(sum).reset_index()


                


            
            
        
        
        
        
        