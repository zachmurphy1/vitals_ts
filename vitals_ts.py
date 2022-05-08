"""
This package provides vitals timeseries processing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simps
import itertools

""" Event handling """
def get_valid_events(events, nested_events=[
    {'name':'ADT', 'start': 'HOSP_ADMSN_TIME', 'end':'HOSP_DISCH_TIME', 'required':True},
    {'name':'Anesthesia', 'start': 'AnesthesiaStartTime', 'end':'AnesthesiaEndTime', 'required':True},
    {'name':'CPB', 'start': 'CPB_start', 'end':'CPB_end', 'required':False},
    {'name':'AXC', 'start': 'AXC_start', 'end':'AXC_end', 'required':False}
]):
    """Determines which events are valid within nested structure
    
    Args:
        events (df): DataFrame with columns
            - id (str): unique ID
            - event (str): name of event (with start/stop)
            - timestamp (datetime64[ns]): timestamp for event
        nested_events (list of dict): ordered list of dicts representing outer to inner events with keys
            - name (str): Name of event (without start/stop)
            - start (str): Name of start event in events df
            - end (str): Name of end event in events df
            - required (bool): Whether event is required to be valid
    """
    
    # Event and flag containers
    events_paired = {}
    flags = {}
    for i in nested_events:
        events_paired[i['name']] = pd.DataFrame()
        flags[i['name']] = []
    
    # Recursively loop through event levels and nested sub-levels
    def get_nested_events(e, level=0):
        # Validate events
        paired, flag = validate_events(e, nested_events[level]['start'], nested_events[level]['end'])
        paired['event'] = nested_events[level]['name']
        if nested_events[level]['required'] & (paired.shape[0] == 0):
            flag = flag + ['CRITICAL_no_{}_events'.format(nested_events[level]['name'])]
        
        # Append events and flags to container
        events_paired[nested_events[level]['name']] = pd.concat([events_paired[nested_events[level]['name']], paired])
        flags[nested_events[level]['name']] = flags[nested_events[level]['name']] + flag
        
        # Loop through each event
        if level < len(nested_events) - 1:
            for i, r in paired.iterrows():
                e_sub = e[(e['timestamp'] >= r['start'])&(e['timestamp'] <= r['end'])]
                get_nested_events(e_sub, level+1)
                
    get_nested_events(events)
    
    # Make events and flags nice
    events_paired_df = pd.DataFrame()
    for i in nested_events:
        events_paired_df = pd.concat([events_paired_df, events_paired[i['name']]])
    events_paired_df.reset_index(inplace=True)
    
    # Return events and flags
    if events_paired_df.shape[0] > 0:
        events_paired_df = events_paired_df[['id','event','event_num','start','end']]
    else:
        events_paired_df = pd.DataFrame().from_dict({'id':[], 'event':[], 'event_num':[],'start':[], 'end':[]})
    return events_paired_df, flags
    
    
def validate_events(df, start_name, end_name, repeat_max_min=5, min_event_duration=1):
    """Pairs and filters events at level
    
    Args:
        - df (df): long-form DataFrame with columns
            - id (str): unique ID
            - event (str): name of event (with start/stop)
            - timestamp (datetime64[ns]): timestamp for event
        - start_name (str): name of start event
        - end_nname (str): name of end event
        - repeat_max_min (float): maximum minutes between repeated start/end events to count them as one (later one gets dropped)
        - min_event_duration (float): minimum duration in minutes for a start/end pair to count as valid
    
    """
    # Get events at level
    df = df[df['event'].isin([start_name, end_name])].copy()
    flags = []
    
    # Separate starts and ends
    starts = df[df['event']==start_name].sort_values('timestamp').reset_index(drop=True)
    ends = df[df['event']==end_name].sort_values('timestamp').reset_index(drop=True)
    
    
    # Remove repeats if less than repeat_max_min minutes apart
    df = pd.concat([starts, ends]).sort_values('timestamp').reset_index(drop=True)
    idx_to_remove = []
    for i in range(df.shape[0]-1):
        if (df.iloc[i]['event'] == df.iloc[i+1]['event']) & ((df.iloc[i+1]['timestamp'] - df.iloc[i]['timestamp'])/np.timedelta64(1,'m') < repeat_max_min):
            idx_to_remove.append(i+1)
    df.drop(df.iloc[idx_to_remove].index, inplace=True)
    if len(idx_to_remove) > 0:
        flags.append('repeats_{}min_removed'.format(repeat_max_min))
    
    # Remove ends that come before first start and starts that come after last end
    if starts.shape[0] > 0:
        if ends[ends['timestamp'] <= starts.iloc[0]['timestamp']].shape[0] > 0:
            flags.append('end_before_start_removed')
        ends = ends[ends['timestamp'] > starts.iloc[0]['timestamp']]
    if ends.shape[0] > 0:
        if starts[starts['timestamp'] >= ends.iloc[-1]['timestamp']].shape[0] > 0:
            flags.append('start_after_end_removed')
        starts = starts[starts['timestamp'] < ends.iloc[-1]['timestamp']]
        
    
    # Get pivot, return empty by default
    pivoted = pd.DataFrame()
    
    # Check if any events 
    if df.shape[0] > 0:
        # Check if equal number of starts and ends
        if (df[df['event']==start_name].shape[0] == df[df['event']==end_name].shape[0]):
            # Check if starts and ends alternate
            if (df['event'] == [start_name, end_name]*(df.shape[0]//2)).all():
                df['event_num'] = np.concatenate([[l]*2 for l in range(df.shape[0]//2)])
                pivoted = df.pivot(index=['id', 'event_num'], columns='event', values='timestamp').rename(columns={start_name:'start', end_name:'end'})
                
                # Remove events with duration < min_event_duration
                if ((pivoted['end'] - pivoted['start'])/np.timedelta64(1,'m') < min_event_duration).any():
                    flags.append('short_duration_removed')
                pivoted = pivoted[(pivoted['end'] - pivoted['start'])/np.timedelta64(1,'m') >= min_event_duration]
            else:
                flags.append('CRITICAL_start_end_not_alternating')
        else:
            flags.append('CRITICAL_start_end_unequal')
    
    return pivoted, flags


def ts_from_events(df, order=['ADT','Anesthesia','CPB','AXC'], freq='1T'):
    """Creates timeseries labelled by df event status
    
    Args:
        - df (df): events_paired DataFrame from get_valid_events
        - order (list of str): Order of event nesting
        - freq (str): frequency arg passed to pd.date_range
        
    Returns:
        - ts (df): DataFrame with below cols
            - index (datetime64): Timestamp
            - Column for each event in 'order' (str): values are pre/intra/post by event number
    """
    # If no events return df with no rows
    if df.shape[0] == 0:
        return pd.DataFrame().from_dict({'timestamp': []}).set_index('timestamp')
    
    # DataFrame containner
    ts = pd.DataFrame().from_dict({'timestamp': pd.date_range(start=df['start'].min(), end=df['end'].max(), freq=freq)}).set_index('timestamp')
    for c in order:
        ts[c] = np.NaN
    event_counts = {}
    for o in order:
        event_counts[o] = 0
        
    # Recursively determine nested pre/intra/post times
    def set_nested_ts(e, start=None, end=None, level=0):     
        if level < len(order):
            events_at_level = e[e['event']==order[level]].reset_index()
            for i in range(events_at_level.shape[0]):
                r = events_at_level.iloc[i]
                # Set timeseries
                ts.loc[(ts.index >= r['start']) & (ts.index <= r['end']), order[level]] = 'intra_{}'.format(event_counts[order[level]])
                
                if (start != None) & (end != None):
                    if i == 0:
                        ts.loc[(ts.index >= start) & (ts.index < r['start']), order[level]] = 'pre_{}'.format(event_counts[order[level]])
                    if i == events_at_level.shape[0]-1:
                        ts.loc[(ts.index > r['end']) & (ts.index <= end), order[level]] = 'post_{}'.format(event_counts[order[level]])
                    if i > 0:
                        prev_end = events_at_level.iloc[i-1]['end']
                        ts.loc[(ts.index > prev_end) & (ts.index < r['start']), order[level]] = 'between_{}_{}'.format(event_counts[order[level]]-1, event_counts[order[level]])
                event_counts[order[level]] += 1
                
                # Get nested
                events_sub = e[(e['start'] >= r['start']) & (e['end'] <= r['end'])]
                set_nested_ts(events_sub, r['start'], r['end'], level+1)
    set_nested_ts(df)
    
    return ts

""" Waveform filtering and interpolation """
def split_bp(df, name, sbp_name='SBP', dbp_name='DBP', remove_orig=True):
    """Splits BP given as sbp/dbp into separate columns
    
    Args:
        df (df): Total dataframe in long form, will be returned modified
        name (str): Name of measure containing BP strings as sbp/dbp
        sbp_name (str): What to name SBP measure
        dbp_name (str): What to name DBP measure
        remove_orig (bool): Whether to remove original name measures from df
        
    Returns:
        df (df): Long form df with sbp and dbp measures appended
    
    """
    
    # Handle SBP
    sbp = df[df['Measure']==name].copy()
    sbp['Value'] = sbp['Value'].apply(lambda x: np.NaN if x != x else x.split('/')[0])
    sbp['Measure'] = sbp_name

    # HAndle DBP
    dbp = df[df['Measure']==name].copy()
    dbp['Value'] = dbp['Value'].apply(lambda x: np.NaN if x != x else x.split('/')[1])
    dbp['Measure'] = dbp_name

    # Concat SBP and DBP, and return
    df = pd.concat([df, sbp, dbp])
    if remove_orig:
        df = df[df['Measure'] != name]
    return df
    
# Standard absolute min/max ranges
abs_filter = {
        'Pulse': (20, 220),
        'mABP': (10, 250),
        'SBP': (20, 300),
        'DBP': (5, 225),
        'Pulse': (30, 150),
        'CVP': (3, 20),
        'RR': (5, 50),
        'Temp': (20, 43),
        'SaO2': (50, 100),
        'FiO2': (21, 100),
        'PEEP': (0, 25),
        'EtCO2': (5, 100),
        'GCS': (0,15),
        'Na': (100,200),
        'K': (1,15),
        'CO2': (3, 60),
        'BUN': (1,200),
        'Cr': (0.1,25),
        'Glc': (10,2000),
        'Albumin': (0.4, 7),
        'TBili': (0,50),
        'Hgb': (1,50),
        'WBC': (0.1, 100),
        'Plt': (1, 2000),
        'INR': (0.5,20),
        'PTT': (10,200),
        'pH': (6.5, 7.8),
        'pO2': (10,700),
        'pCO2': (10,200),
        'HCO3': (5, 100),
        'BE': (-30, 30)
    }

def absolute_filter(dat, abs_filter=abs_filter):
    """Removes values outsize min/max ranges.
    
    """
    for k, v in abs_filter.items():
        if k in dat.columns:
            dat.loc[(dat[k] < v[0])|(dat[k] > v[1]), k] = np.NaN 
    return dat

def spike_filter(df, cols, thresh={'SBP':100, 'mABP':100, 'DBP':100}, method='slope-diff'):
    """Remove lab spikes in a-line data.
    
    Args:
        df (df): DataFrame with timestamp index and measures in columns
        cols (list of str): Columns to remove lab spikes from
        thresh (dict): Threshold for each col
        method (str): Which method to use to identify lab spikes
            - slope-diff: Calculate difference between slope coming into and going out of each point, supposes that lab spikes will have high positive slope going nto the point annd high negative slope coming out of point.
    
    """
    if method == 'slope-diff':
        for c in cols:
            df_sub = df[c].copy().dropna()
            spikes = df_sub.diff(1).diff(-1)
            spikes_idx = spikes[spikes >= thresh[c]].index.tolist()
            df.loc[spikes_idx, c] = np.NaN
    return df
    
def interpolate(df, cols, point_valid_time=5, how='linear'):
    """Interpolates columns to frequency of df
    
    Args:
        df (df): DataFrame with timeseries index, measures in cols
        cols (df): Which cols to interpolate
        point_valid_time (float): Minutes to interpolate gaps
        how (str): Interpolationn method from pandas.interpolate
    
    """
    for c in cols:
        df[c] = df[c].interpolate(method=how, limit=point_valid_time, limit_area='inside')
    return df
    

""" Feature calculation """
def add_feature(feat, to_add):
    """Top-level funcntion for adding features"""
    return {**to_add, **feat}

def feature_by_absolute_time(df, cols, func, name, start, end, **kwargs):
    """Calculate a feature by absolute start/end times
    
    Args:
        - df (df): DataFrame with timestamp index and measures in cols
        - cols (list of str): cols to get feature on
        - func (function): function that returns calculated feature
        - name (str): name of feature to use in resulting feature columnn
        - start (datetime64): Start timestamp (inclusive)
        - end (datetime64): End timestamp (not innclusive
        - **kwargs (dict): args passed to func)
        
    Returns:
        features (dict): Returns dict of feature values
    
    """
    features = {}
    df_sub = df.loc[(df.index >= start) & (df.index < end)]
    
    for c in cols:
        features['{}__{}'.format(c, name)] = func(df_sub[c], **kwargs)
        
    return features

def feature_by_event_time(df, cols, events, func, name, abs_timeframe_min=None, abs_timeframe_from=None, **kwargs):
    """Calculate a feature by event pre/intra/post times
    
    Args:
        - df (df): DataFrame with timestamp index and measures in cols
        - cols (list of str): cols to get feature on
        - events (df): DataFrame of events from get_valid_events
        - func (function): function that returns calculated feature
        - name (str): name of feature to use in resulting feature columnn
        - abs_timeframe_min (str): Minutes to get in subevent
        - abs_timeframe_from (str): Whether to take subevetn from start or end (None, 'start', or 'end')
        - **kwargs (dict): args passed to func)
    
    
    """
    features = {}
    for e in events:
        events_at_level = df[e].drop_duplicates().dropna().tolist()
        for e_level in [[l_] for l_ in events_at_level] + [list(l_) for l_ in itertools.combinations(events_at_level, 2)]:
            
            df_sub = df[df[e].isin(e_level)]
            if abs_timeframe_min != None:
                if abs_timeframe_from == 'start':
                    df_sub = df_sub.loc[df_sub.index < df_sub.index.min() + pd.Timedelta(minutes=abs_timeframe_min)]
                elif abs_timeframe_from == 'end':
                    df_sub = df_sub.loc[df_sub.index > df_sub.index.max() - pd.Timedelta(minutes=abs_timeframe_min)]
                else:
                    raise ValueError('abs_timeframe_from must be start or end')
            for c in cols:
                if df_sub[c].dropna().shape[0] > 0:
                    features['{}__{}__{}__{}'.format(e, '^'.join(e_level), c, name)] = func(df_sub[c], **kwargs)
                else:
                    features['{}__{}__{}__{}'.format(e, '^'.join(e_level), c, name)] = np.NaN
    return features

def ts_min(x, **kwargs):
    return x.min()

def ts_argmin(x, **kwargs):
    return x.argmin()

def ts_max(x, **kwargs):
    return x.max()

def ts_argmax(x, **kwargs):
    return x.argmax()

def ts_first(x, **kwargs):
    return x.dropna().iloc[0]

def ts_last(x, **kwargs):
    return x.dropna().iloc[-1]

def ts_argfirst(x, **kwargs):
    return x.dropna().index[0]

def ts_arglast(x, **kwargs):
    return x.dropna().index[-1]

def ts_total_time(x, **kwargs):
    return x.shape[0]

def ts_valid_time(x, **kwargs):
    return (~x.isna()).sum()
    
def ts_mean(x, **kwargs):
    return np.nanmean(x)

def ts_tut(x, **kwargs):
    """Get time under/above threshold
    
    Args:
        - x (Series): Timestamp-indexed Series to calculate feature
        - how (str): Whether to get under/above threshold
        - inclusive (bool): Whether to include threshold in calculation
        - thresh (float): Threshold
        
    Returns:
        - (float): Area as defined by args
    
    """
    
    if kwargs['how'] == 'under':
        if kwargs['inclusive']:
            return (x <= kwargs['thresh']).sum()
        else:
            return (x < kwargs['thresh']).sum()
    elif kwargs['how'] == 'above':
        if kwargs['inclusive']:
            return (x >= kwargs['thresh']).sum()
        else:
            return (x > kwargs['thresh']).sum()
    
def ts_aut(x, **kwargs):
    """Get area under/above threshold
    
    Args:
        - x (Series): Timestamp-indexed Series to calculate feature
        - how (str): Whether to get under/above threshold
        - inclusive (bool): Whether to include threshold in calculation
        - thresh (float): Threshold
        
    Returns:
        - aut (float): Area as defined by args
    
    """
    # Adjust to thresh, invert, make NaN if >= thresh
    if kwargs['how'] == 'under':
        if kwargs['inclusive']:
            x = x.apply(lambda v: np.NaN if v > kwargs['thresh'] else kwargs['thresh'] - v)
        else:
            x = x.apply(lambda v: np.NaN if v >= kwargs['thresh'] else kwargs['thresh'] - v)
            
    elif kwargs['how'] == 'above':
        if kwargs['inclusive']:
            x = x.apply(lambda v: np.NaN if v < kwargs['thresh'] else v - kwargs['thresh'])
        else:
            x = x.apply(lambda v: np.NaN if v <= kwargs['thresh'] else v - kwargs['thresh'])
    else:
        raise ValueError('how must be \'above\' or \'under\'')
    
    # Break into discrete pieces separated by NaN
    x_discrete = x.copy()
    x_discrete.reset_index(drop=True, inplace=True)
    aut = 0
    while x_discrete.shape[0] > 0:
        # Get start and slide to start
        start = x_discrete[~x_discrete.isna()].index.min()
        if np.isnan(start):
            break
        x_discrete = x_discrete.loc[start:]
        
        # Get stop, make piece, slide x_discrete to stop
        stop = x_discrete[x_discrete.isna()].index.min()
        if np.isnan(stop):
            x_piece = x_discrete
        else:
            x_piece = x_discrete.loc[:stop-1]
        x_discrete = x_discrete.loc[stop:]
        
        # Get area
        if kwargs['integrator'] == 'left-riemann':
            a = x_piece.sum()
        elif kwargs['integrator'] == 'simpson':
            a = simps(x_piece, x_piece.index)
            a += x_piece.iloc[-1]
        elif kwargs['integrator'] == 'trapezoid':
            a = np.trapz(x_piece, x_piece.index)
            a += x_piece.iloc[-1]
        else:
            raise ValueError('integrator must be ***')
        
        aut += a
        
    return aut
        
""" Visualization """
def plot_ts(dat, cols, discrete=[], events=True, events_list=None, figsize=(20,7), event_limit=None, connect_na_lines=False):
    fig, ax = plt.subplots(2,1, figsize=figsize)
    for c in cols:
        # Connect NA lines arg
        if connect_na_lines:
            dat_sub = dat[~dat[c].isna()].copy()
        else:
             dat_sub = dat.copy()
                
        # Plot
        if c in discrete:
            ax[0].scatter(dat_sub.index, dat_sub[c], label=c)
        else:
            ax[0].plot(dat_sub.index, dat_sub[c], label=c)
    ax[0].legend()
    
    for e in events_list:
        event_sub = dat[~dat[e].isna()].copy()
        event_sub[e] = event_sub[e].apply(lambda x: '{}_{}'.format(e,x))
        ax[1].scatter(event_sub.index, event_sub[e])
    
    if event_limit:
        if type(event_limit) == str:
            lim_df = dat[dat[event_limit].apply(lambda x: 'intra' in str(x))]
            if lim_df.shape[0] > 0:
                ax[0].set_xlim(lim_df.index[0], lim_df.index[-1])
                ax[1].set_xlim(lim_df.index[0], lim_df.index[-1])
        elif type(event_limit) == tuple:
            ax[0].set_xlim(event_limit[0], event_limit[1])
            ax[1].set_xlim(event_limit[0], event_limit[1])
    plt.show()
