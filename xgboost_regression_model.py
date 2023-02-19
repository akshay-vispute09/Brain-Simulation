
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import pickle

def model_building(data):

  data.drop(['Unnamed: 0'], axis = 1, inplace = True)

  # Seperating dependent and independent variable
  X = data.drop(['sine_freq', 'sine_amp', 'sine_phase'], axis= 1)
  y = data[['sine_freq', 'sine_amp', 'sine_phase']]

  # Splitting the data randomly, with 20% data in test set
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

  # converting data in to numpy array ( to resolve issues while inferencing)
  X_train = X_train.to_numpy()
  X_test = X_test.to_numpy()

  # fitting Multi output regressor , this will generate 3 output per input 
  multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror')).fit(X_train, y_train)

  preds = multioutputregressor.predict(X_test)

  # convert test target variables to numpy array
  y_test_arr = y_test.to_numpy()

  output_data = np.concatenate((y_test_arr, preds), axis= 1)
  test_output = pd.DataFrame({'sine_freq': output_data[:, 0], 'sine_amp': output_data[:, 1], 'sine_phase': output_data[:, 2],
                'sine_freq_pred': output_data[:, 3], 'sine_amp_pred': output_data[:, 4], 'sine_phase_pred': output_data[:, 5]})



  pickle.dump(multioutputregressor,f)


  # ## Loading and Inferencing

  # In[19]:


  import pickle
  # load pickle file
  with open('model.pkl', 'rb') as f:
      reg = pickle.load(f)


  # In[20]:


  def get_predictions(input_arr):
    if input_arr.shape != (1,1024): 
      input_arr = input_arr.reshape(1,1024)
    
    return reg.predict(input_arr).flatten()


  # In[ ]:





  # In[25]:


  # load data

  import pandas as pd
  data = pd.read_csv('training_set.csv')
  data.drop(['Unnamed: 0'], axis = 1, inplace = True)
  X = data.drop(['sine_freq', 'sine_amp', 'sine_phase'], axis= 1)
  y = data[['sine_freq', 'sine_amp', 'sine_phase']]


  # In[26]:


  X[-2:-1]


  # In[27]:


  # predict the output

  sine_freq, sine_amp, sine_phase = get_predictions(X[-2:-1].to_numpy())
  print(f'Freq : {sine_freq}, Amp : {sine_amp}, Phase : {sine_phase}')


  # In[28]:


  # actual output
  y[:-2:-1]


  # In[ ]:




