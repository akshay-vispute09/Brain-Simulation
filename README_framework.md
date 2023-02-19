## Closed Loop Therapeutic Brain Simulation

The mental health has been a subject of matter from decades. The
department of neuroscience is developing several treatments in order to
overcome this problem.The proposed solution focused on exploring an EEG
(Electrocencephalography) data and developing a system that can help
overcome the mental disorders.

Steps to Run: 1) Make sure the MNE packages are installed and run the
following command ,

    import mne

2)  Consider the following EEG data (Somatosensory data) which is
    already present in the MNE package. Load the data using
    mne.io.read\_raw\_fif() function,

   raw\_path = mne.datasets.somato.data\_path() +
    '/sub-01/meg/sub-01\_task-somato\_meg.fif' raw =
    mne.io.read\_raw\_fif(raw\_path)

3)  Determine the number of channels present in the data using the
    following command,

   raw.info.get('nchan')

4)  Exploratory Data Analysis is an important step have a broader look
    on the data . Run the following command to have a visualisation on
    the raw EEG data,

   raw.plot()

5)  In order to get a broader view on the insights of the EEG signals
    regarding the frequency, channel names etc the following command can
    be used,

  print(raw) 
  print(raw.info)



#### Steps to run.

    1.Reuired libraries should be installed and Imported.
    2.It is a python file and should be ready to go on any IDE for python3.

#### Generating Sine waves with noice (generate_sine)
    Function: generate_sine
    Input: None
    Output: a dictionary with 3 key value pairs
    key 1: df
    Value:a dataframe with 6 columns 'timestamp', 'channe_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5'
    Note: timestamps are indices of the samples (1024 x 5 = 5120 in this case for 5 seconds of data in each channel)
    key 2: samp_freq 
    value: number of samples taken in a second (1024 in this case)
    key 3: duration
    value: the time length of data ( 5 seconds in this case)

#### Importing real EEG data (get_real)
    Input: csv file with real data
    Output: A dictionary that returns the dataframe, sampling frequency and length of data in seconds.

#### Creating windows (window_maker)
    function: window_maker
    input: dataframe, sample frequency, duration
    output: A dictionary with 5 keys, each representing a channel(c1, c2, c3, c4 and c5). Each channel is a list of 5 lists, each list containing data for one second(5 lists ~ 5 seconds).
    
#### Fitting sine and predicting amplitude, anfular frequency, phase,offset, frequency and period (fit_sin)
    Input: Array and timestamps, both in for of lists or numpy arrays
    Outpur: A dictionary with the values of aplitude,angular frequency, phase, offset, frequency and period.
    Output keys and values:
    1.'amp' is the amplitude of sine wave that os fit to the data.
    2.'omega' is the angulatr frequency.
    3.'phase' is the phase.
    4.'offset' is the offset.
    5.'freq' is the frequency.
    6.'period' is the period.
    
#### Using linear regression to predict phase, frequency and amplitude (getPrediction_linear)
    Input: datapoints.
    Output: predicted frequency, amplitude and phase.  
    
#### Using XG Boost to predict phase, frequency and amplitude (getPrediction_linear)
    Input: datapoints.
    Output: predicted frequency, amplitude and phase.  
    
#### Function to decide which method to be used for predictio (sine_attr)
    Input: timestamps, datapoints and the method for prediction.
    Output: Predicted amplitude, angular frequency and phase of sine wave.
    
#### Function to evaluate the prediction (evaluate)
    Input: 2 arrays, ground truth and the prediction
    Output: prints Mean Absolute Error, Mean Square Error, Root Mean Square Error, R squared score and Similarity Score
        
#### Function to fplot prediction vs truth(display)
    Input: data points
    Output: visualization of second-by-second prediction constructed sine wave against ground truth. Average RMSE, MAE, MSE and R-squred values over the course of prediction.


## real-time_normalized_data.py
This file takes the MNE sample dataset as input, calculates the sampling frequency, filters the alpha signal, normalizes the mean for every channel to 0, and returns the sampling frequency and normalized dataframe

## simulated_data_generation.py
This file creates a simulated data frame where the sampling frequency is provided as input and it will give simulated dataset

## linear_regression_model.py
This file is used to pre-train the linear regression model so that it can be used for the prediction of phase and frequency on any dataset

## xgboost_regression_model.py
This file is used to pre-train xgboost regression model so that it can be used for the prediction of phase and frequency on any dataset