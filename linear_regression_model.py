import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

import normalize
import simulated_data_generation

# Generate ML models
def model_building(df):

    # prediction model for frequency
    x = df.drop(['sine_freq','sine_amp','sine_phase'], axis=1)
    y = df[['sine_freq']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model = LinearRegression(fit_intercept=True,normalize=False,positive=False)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    predictions = model.predict(X_test)

    pickle.dump(model, open('LR_freq.pkl', 'wb'))

    # prediction model for phase
    x = df.drop(['sine_freq','sine_amp','sine_phase'], axis=1)
    y = df[['sine_phase']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model = LinearRegression(fit_intercept=True,normalize=False,positive=False)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    predictions = model.predict(X_test)

    pickle.dump(model, open('LR_phase.pkl', 'wb'))

    # prediction model for amplitude
    x = df.drop(['sine_freq','sine_amp','sine_phase'], axis=1)
    y = df[['sine_amp']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model = LinearRegression(fit_intercept=True,normalize=False,positive=False)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    predictions = model.predict(X_test)

    pickle.dump(model, open('LR_amp.pkl', 'wb'))


if __name__ == "__main__":
    model_building(training_dataset)
