import numpy as np
import pandas as pd
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from random import randrange

# https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
# https://github.com/statsmodels/statsmodels/issues/3503

def sin_additive_demo():
    n_counts = 1000
    n_periods = 10
    freq = int(n_counts/n_periods) # number of counts per one period

    x = np.linspace(0, 2*np.pi * n_periods, n_counts)
    y = np.sin(x) + x/10 + np.random.randn(n_counts)*0.2

    dec = seasonal_decompose(y, model='additive', freq=freq)
    dec.plot()
    pyplot.show()


def preprocess_data(data):
    data.info(verbose=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'])    
    data.set_index('timestamp', inplace=True)
    data.sort_index(inplace=True)
    print(data.head(20))
    return data


if __name__ == '__main__':
    
    file = r'..\..\Data\alerts.csv'
    data = pd.read_csv(file)
    
    data = preprocess_data(data)
    
    pyplot.plot(data.index, data.values)
    pyplot.grid(True)
    pyplot.show()

    n_counts = len(data.index)
    n_periods = n_counts / 24 / 4 # seasonality - 1 day; 24 * 4 - number of counts per day
    freq = int(n_counts/n_periods) # number of counts per one period

    av_data = pd.ewma(data, span=(24*4))
    pyplot.plot(av_data.index, av_data.values)
    pyplot.grid(True)
    pyplot.show()


    #fft = np.fft.fft(data["eventCount"].values)
    #pyplot.plot(data.index, fft)
    #pyplot.grid(True)
    #pyplot.show()

    #dec = seasonal_decompose(data["eventCount"].values, model='multiplicative', freq=freq) # multiplicative? additive?

    #pyplot.plot(data.iloc[1100:1295, :].index, dec.resid[1100:1295])
    #pyplot.grid(True)
    #pyplot.show()

    #print(dec.trend[1277:1279])
    #print(data.iloc[1277:1279, :])

    #dec.plot()
    #pyplot.grid(True)
    #pyplot.show()

    
