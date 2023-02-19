import pandas as pd
import numpy as np
import mne

def normalized_data_generation():

    # loading the data
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = (sample_data_folder / 'MEG' / 'sample' /
                        'sample_audvis_filt-0-40_raw.fif')

    # reading raw channel
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload = True, verbose=False)

    # storing sampling frequency
    sampling_frequency = round(raw.info.get('sfreq'))

    sample_data_events_file = (sample_data_folder / 'MEG' / 'sample' /
                            'sample_audvis_filt-0-40_raw-eve.fif')
    events = mne.read_events(sample_data_events_file)

    event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
                'visual/right': 4}

    low_cut = 8
    hi_cut  = 12 

    # Filtering alpha signal
    raw_filt = raw.copy().filter(low_cut, hi_cut)

    # converting raw channel to dataframe
    dataframe = raw_filt.to_data_frame()

    # creating a new dataframe
    new_dataframe = pd.DataFrame()

    # copying time column to new dataframe
    new_dataframe['time'] = dataframe['time'].copy()

    # Copying columns that strats with "EEG"
    new_dataframe = new_dataframe.join(dataframe.loc[:, dataframe.columns.str.startswith('EEG')])

    # normalizing mean for every column to 0 and stoting in new dataframe
    normalized_dataframe= (new_dataframe - new_dataframe.mean()) / (new_dataframe.std())
    
    return sampling_frequency, normalized_dataframe

# if __name__ == "__main__":
#     normalized_data_generation()