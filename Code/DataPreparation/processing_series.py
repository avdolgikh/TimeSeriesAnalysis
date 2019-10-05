import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa as tsa
import matplotlib.pyplot as plt
from scipy.special import boxcox1p


def plot(index, series, title=""):
    fig = plt.figure(figsize=(6, 5), dpi=300)
    plt.plot( index, series, linewidth=1)
    plt.title(title)
    plt.grid(True)
    plt.show()
    return fig

def dfuller(series):
    # check whether it is stationary
    return sm.tsa.stattools.adfuller(series)

def preprocess(data, value_field):
    data.sort_index(inplace=True)
    data[value_field] = data[value_field].astype(np.float32)

def convert_2_daily_values(data, value_fiel):
    for row_index, row in data.iterrows():   
        days_in_month = pd.Period( str(row_index) ).days_in_month
        row[value_fiel] = row[value_fiel] / days_in_month

if __name__ == '__main__':
    milk = pd.read_csv(r'..\..\Data\monthly-milk-production.csv', ';', index_col=['month'], parse_dates=['month'], dayfirst=True)
    preprocess(milk, "milk")
    convert_2_daily_values(milk, "milk")
    plot(milk.index, milk["milk"])
    print(milk["milk"].iloc[:10])    

    dfull = dfuller(milk["milk"])
    print(dfull[1]) # check whether dfull[1] < 0.2 ???

    # 1. Make the series stationary
    
    # 1.1. Stabilization of Variation
    #milk["milk"] = milk["milk"].apply(np.log)
    milk["milk"] = boxcox1p(milk["milk"], 0.25) # from scipy import stats, stats.boxcox(data)
    # lambda = 0.25?
    #print(milk["milk"].iloc[:10])
    #plot(milk, "milk")

    # 1.2. Differencing the series
    # https://machinelearningmastery.com/difference-time-series-dataset-python/
    # https://machinelearningmastery.com/remove-trends-seasonality-difference-transform-python/
    # 1.2.1. Season diff
    milk["milk"] = milk["milk"].diff(periods=12)
    # 1.2.2. diff
    milk["milk"] = milk["milk"].diff()
    #print(milk["milk"].iloc[:30])
    milk = milk.iloc[13:, :]
    #print(milk["milk"].iloc[:30])
    dfull = dfuller(milk["milk"])
    print(dfull[1])
    plot(milk.index, milk["milk"])

    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_model.ARMA.html
    arma = tsa.arima_model.ARMA(milk["milk"], order=(2, 2))
    print(arma.fit())

    # https://www.coursera.org/learn/data-analysis-applications/lecture/8yR4G/arima
    


    # Autocorrelation, # https://www.coursera.org/learn/data-analysis-applications/lecture/4PEHZ/avtokorrieliatsiia
    #acf = sm.tsa.stattools.acf(milk["milk"], nlags=10)
    #print(acf)

    # Correlograms:
    # https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html
    plot_acf(milk["milk"], lags=50)
    plt.grid(True)
    plt.show()
    plot_pacf(milk["milk"], lags=50)
    plt.grid(True)
    plt.show()

    
    
    