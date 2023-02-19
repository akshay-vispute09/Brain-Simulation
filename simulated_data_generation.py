import pandas as pd
import numpy as np 
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def data_generation(sampling_frequency):

    training_dict = {}

    for i in range(sampling_frequency):
        col_name = 'T_'+str(i)
        training_dict[col_name] = [0]
        
    df = pd.DataFrame(training_dict)
    df=df.drop(df.index[0])

    sine_freq_list = [8,9,10,11,12]
    sine_amp_list = list(range(5,25))
    sine_phase_list = [np.pi/2, np.pi/3, np.pi/4, np.pi/5]
    x = np.linspace(0,1, sampling_frequency)

    freq_l = []
    amp_l = []
    phase_l = []

    for freq in sine_freq_list:
        for amp in sine_amp_list:
            for phase in sine_phase_list:
                y = np.sin(2*np.pi*freq* x + phase) * amp
                temp_dict = {}
                for i in range(sampling_frequency):
                    temp_dict['T_'+str(i)] = y[i]
                freq_l.append(freq)
                amp_l.append(amp)
                phase_l.append(phase)    
                df = df.append(temp_dict, ignore_index=True)

    noice_level = 5         
    cols = list(df.columns)
    for ch in cols:
        df[ch] = df[ch].apply(lambda x: x + random.uniform(-noice_level,noice_level))

    df['sine_freq'] = freq_l
    df['sine_amp'] = amp_l
    df['sine_phase'] = phase_l
    return df