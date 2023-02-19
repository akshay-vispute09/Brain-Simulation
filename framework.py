import numpy as np
import random
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from sklearn import metrics
import pickle
import xgboost
import mne

import normalize
import simulated_data_generation
import linear_regression_model

sample_freq, normalized_dataframe = normalize.normalized_data_generation()
training_dataset =  simulated_data_generation.data_generation(sample_freq)
linear_regression_model.model_building(training_dataset)

# to avoid throwing unnecessary error
plt.rcParams.update({'figure.max_open_warning': 0})

# importing pretrained ml models using pickle
pkl_filename = "LR_freq.pkl"
with open(pkl_filename, 'rb') as file:
    freq_model = pickle.load(file)
pkl_filename = "LR_amp.pkl"
with open(pkl_filename, 'rb') as file:
    amp_model = pickle.load(file)
pkl_filename = "LR_phase.pkl"
with open(pkl_filename, 'rb') as file:
    phase_model = pickle.load(file)

pkl_filename = "xg_freq.pkl"
with open(pkl_filename, 'rb') as file:
    xg_frequency = pickle.load(file)
pkl_filename = "xg_amp.pkl"
with open(pkl_filename, 'rb') as file:
    xg_amplitude = pickle.load(file)
pkl_filename = "xg_phase.pkl"
with open(pkl_filename, 'rb') as file:
    xg_phase = pickle.load(file)



# This function generates noisy sine waves that can be used as input.
# input: None
# output: A dictionary that returns the dataframe, sampling frequency and length of data in seconds.
def generate_sine():
    samp_freq = sample_freq
    duration = 5  # for creating data for 5 seconds
    phase_list = [1, 2, 3, 4, 5]
    freq_list = [8, 9, 11, 10, 8]
    amp_list = [1, 2, 3, 4, 5]
    # Multi channel sine generation
    sine_list = []
    x = np.linspace(0, 1, samp_freq)
    dic = {'index': list(range(samp_freq * duration))}
    t_stamps = dic['index']
    # creating sine values
    for i in range(5):
        y = np.sin(2 * np.pi * freq_list[i] * x + phase_list[i]) * amp_list[i]
        y = list(y) * duration
        sine_list.append(y)
        col = "channel_" + str(i + 1)
        dic[col] = y
    # noice introduction
    df = pd.DataFrame(dic)
    df.drop('index', axis=1, inplace=True)
    # noice_level = float(input("Enter a noice level between 0 and 1 : "))
    noice_level = .5
    # introducing noice
    cols = list(df.columns)
    for ch in cols:
        df[ch] = df[ch].apply(lambda val: val + random.uniform(-noice_level, noice_level))
    # inserting timestamps
    df.insert(0, 'timestamps', pd.Series(t_stamps), True)

    return {'df': df, 'samp_freq': samp_freq, 'duration': duration}


# This function breaks down data into chunks equivalent of 1 second for second by second prediction.
# input: A dictionary containing a dataframe, sampling frequency and length of data in seconds.
# output: A dictionary with second by second data for 5 channels, sampling frequency and length of data in seconds.
def window_maker():

    # generate_sine function is used to create dummy data for this demo which are stored in dictionary 'df_d'
    df_d = generate_sine()
    # extracting dataframe, sampling frequency and duration from the dictionary
    df = df_d['df']  # dataframe
    sampling_frequency = df_d['samp_freq']  # sampling frequency
    duration = df_d['duration']  # length of data in seconds
    d1, d2, d3, d4, d5 = df.channel_1, df.channel_2, df.channel_3, df.channel_4, df.channel_5
    d_list = [d1, d2, d3, d4, d5]
    l1, l2, l3, l4, l5 = [], [], [], [], []
    # l1 will have data of channel_1 in 5 windows(lists), representing each second of 5 seconds of data
    l_list = [l1, l2, l3, l4, l5]
    for m in range(5):
        d = d_list[m]
        ll = l_list[m]
        temp = []
        count = 0
        for i in range(sampling_frequency * duration):
            temp.append(d[i])
            count += 1
            if count == sampling_frequency:
                count = 0
                ll.append(temp)
                temp = []
    return [{'c1': l1, 'c2': l2, 'c3': l3, 'c4': l4, 'c5': l5}, sampling_frequency, duration]


# This function fits sine into the given data and returns sine variables.
# input: datapoints and timestamps.
# output: amplitude, angular frequency,phase, offset, frequency and period of the fitted sine wave.
def fit_sin(tt, yy):
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
    def sinefunction(t, Amp, Ang_freq, ph, offset): return Amp * np.sin(Ang_freq * t + ph) + offset
    popt, pcov = scipy.optimize.curve_fit(sinefunction, tt, yy, p0=guess, maxfev=100000)
    a, w, p, c = popt
    f = w/(2.*np.pi)
    return {"amp": a, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f}


# This function is used to print the evaluation metrics.
# input: ground truth, prediction and a variable to indicate true after the very last prediction.
# output: average MAE,MSE,RMSE and R-squared values.
def evaluate(a, b, c):
    MAE_list = []
    MSE_list = []
    RMSE_list = []
    R2_list = []
    MAE_list.append(metrics.mean_absolute_error(a, b))
    MSE_list.append(metrics.mean_squared_error(a, b))
    RMSE_list.append(np.sqrt(metrics.mean_squared_error(a, b)))
    R2_list.append(metrics.r2_score(a, b))
    if c == 0:
        print("Average_MAE : ", sum(MAE_list)/len(MAE_list))
        print("Average_MSE : ", sum(MSE_list) / len(MSE_list))
        print("Average_RMSE : ", sum(RMSE_list) / len(RMSE_list))
        print("Average_R_squared : ", sum(R2_list) / len(R2_list))


# This function is used to predict sine variables using pretrained linear regressor.
# input: datapoints.
# output: predicted frequency, amplitude and phase.
def getPrediction_linear(inputData):

    column_names = []

    for i in range(0,sample_freq):
        column_names.append("T_"+str(i))

    dataframe = pd.DataFrame(inputData, columns = column_names)
    freq_pred = freq_model.predict(dataframe)
    amp_pred = amp_model.predict(dataframe)
    phase_pred = phase_model.predict(dataframe)
    return freq_pred, amp_pred, phase_pred


# This function is used to predict sine variables using pretrained XGBoost model.
# input: datapoints.
# output: predicted frequency, amplitude and phase.
def getPrediction_xgboost(inputData):

    column_names = []

    for i in range(0,sample_freq):
        column_names.append("T_"+str(i))
        
    dataframe = pd.DataFrame(inputData, columns = column_names)
    freq_pred = xg_frequency.predict(dataframe)
    amp_pred = xg_amplitude.predict(dataframe)
    phase_pred = xg_phase.predict(dataframe)
    return freq_pred, amp_pred, phase_pred


# This function picks prediction model based on user input.
# input: timestamps, datapoints and the method for prediction.
# output: Predicted amplitude, angular frequency and phase of sine wave.
def sine_attr(timestamps, fit, mode):
    if mode == 'sf':
        sin_attr = fit_sin(timestamps, fit)
        return {'amp': sin_attr['amp'], 'omega': sin_attr['omega'], 'phase': sin_attr['phase']}
    elif mode == 'lr':
        fit = np.array([fit])
        sin_attr = {}
        n, o, p = getPrediction_linear(fit)
        sin_attr['amp'] = o[0]
        sin_attr['omega'] = 2 * np.pi * n[0]
        sin_attr['phase'] = p[0]
        return {'amp': sin_attr['amp'], 'omega': sin_attr['omega'], 'phase': sin_attr['phase']}
    elif mode == 'xg':
        fit = np.array([fit])
        sin_attr = {}
        n, o, p = getPrediction_xgboost(fit)
        sin_attr['amp'] = o[0]
        sin_attr['omega'] = 2 * np.pi * n[0]
        sin_attr['phase'] = p[0]
        return {'amp': sin_attr['amp'], 'omega': sin_attr['omega'], 'phase': sin_attr['phase']}


# This function plots prediction against ground truth after every second of prediction.
# input: data points
# output: visualization of second-by-second prediction constructed sine wave against ground truth. Average RMSE, MAE,
# MSE and R-squred values over the course of prediction.
def display(item):
    global d, sf, dur, tt, ind, c1, c2, c3, c4, c5, this_second_index, next_second_index
    global cp1, cp2, cp3, cp4, cp5, mode
    mode = input("enter \'sf\' for sine fitting, \'lr\' for linear regression or \'xg\' for xgboost algorithm to predict sine variables: ")
    d = item[0]
    sf = item[1]
    dur = item[2]
    tt = np.linspace(0, 1, sf)  # timestamps for data
    c1, c2, c3, c4, c5 = [], [], [], [], []
    cp1, cp2, cp3, cp4, cp5 = [], [], [], [], []
    ind = 0
    this_second_index = list(range(sf))
    next_second_index = list(range(sf,sf+sf))

    def animate(i):
        global d, sf, dur, tt, ind, c1, c2, c3, c4, c5, this_second_index, next_second_index
        global cp1, cp2, cp3, cp4, cp5, mode
        plt.figure()
        for key in d.keys():
            plt.figure()
            if key == 'c1':
                ax1.cla()
                for c in d[key][ind]:
                    c1.append(c)
                ax1.plot(this_second_index, c1)
                fit = np.array(d[key][ind])
                sin_attr = sine_attr(tt, fit, mode)
                new_pred = sin_attr['amp'] * np.sin(sin_attr['omega'] * tt + sin_attr['phase'])
                for cp in list(new_pred):
                    cp1.append(cp)
                ax1.plot(next_second_index, cp1)
                ax1.set_title('channel 1')
                ax1.set_xlabel('timestamps')
                ax1.set_ylabel('amplitude')
                if ind == dur-1:
                    evaluate(c1[sf:], cp1[:-sf], 1)

            if key == 'c2':
                ax2.cla()
                for c in d[key][ind]:
                    c2.append(c)
                ax2.plot(this_second_index, c2)
                fit = np.array(d[key][ind])
                sin_attr = sine_attr(tt, fit, mode)
                new_pred = (sin_attr['amp'] * np.sin(sin_attr['omega'] * tt + sin_attr['phase']))*1
                for cp in list(new_pred):
                    cp2.append(cp)
                ax2.plot(next_second_index, cp2)
                ax2.set_title('channel 2')
                ax2.set_xlabel('timestamps')
                ax2.set_ylabel('amplitude')
                if ind == dur-1:
                    evaluate(c2[sf:], cp2[:-sf], 1)

            if key == 'c3':
                ax3.cla()
                for c in d[key][ind]:
                    c3.append(c)
                ax3.plot(this_second_index, c3)
                fit = np.array(d[key][ind])
                sin_attr = sine_attr(tt, fit, mode)
                new_pred = (sin_attr['amp'] * np.sin(sin_attr['omega'] * tt + sin_attr['phase']))*1
                for cp in list(new_pred):
                    cp3.append(cp)
                ax3.plot(next_second_index, cp3)
                ax3.set_title('channel 3')
                ax3.set_xlabel('timestamps')
                ax3.set_ylabel('amplitude')
                if ind == dur-1:
                    evaluate(c3[sf:], cp3[:-sf], 1)

            if key == 'c4':
                ax4.cla()
                for c in d[key][ind]:
                     c4.append(c)
                ax4.plot(this_second_index, c4)
                fit = np.array(d[key][ind])
                sin_attr = sine_attr(tt, fit, mode)
                new_pred = (sin_attr['amp'] * np.sin(sin_attr['omega'] * tt + sin_attr['phase']))*1
                for cp in list(new_pred):
                 cp4.append(cp)
                ax4.plot(next_second_index, cp4)
                ax4.set_title('channel 4')
                ax4.set_xlabel('timestamps')
                ax4.set_ylabel('amplitude')
                if ind == dur-1:
                 evaluate(c4[sf:], cp4[:-sf], 1)

            if key == 'c5':
                ax5.cla()
                for c in d[key][ind]:
                    c5.append(c)
                ax5.plot(this_second_index, c5)
                fit = np.array(d[key][ind])
                sin_attr = sine_attr(tt, fit, mode)
                new_pred = sin_attr['amp'] * np.sin(sin_attr['omega'] * tt + sin_attr['phase'])
                for cp in list(new_pred):
                    cp5.append(cp)
                ax5.plot(next_second_index, cp5)
                ax5.set_title('channel 5')
                ax5.set_xlabel('timestamps')
                ax5.set_ylabel('amplitude')
                if ind == dur-1:
                    evaluate(c5[sf:], cp5[:-sf], 0)

                for f in range(sf):
                    this_second_index.append(this_second_index[-1] + 1)
                for f in range(sf):
                    next_second_index.append(next_second_index[-1] + 1)

                ind += 1
                if ind >= dur:
                    ani.event_source.stop()

    ani = FuncAnimation(plt.gcf(), animate, interval=1000, repeat=False)
    plt.tight_layout()
    plt.show()


# The main function
if __name__ == "__main__":
    windows = window_maker()
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5, 1)
    display(windows)
