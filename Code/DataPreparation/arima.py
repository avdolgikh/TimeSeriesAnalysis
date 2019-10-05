# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# https://www.coursera.org/learn/data-analysis-applications/lecture/8yR4G/arima

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

def preprocess(data, value_field):
    data.sort_index(inplace=True)
    data[value_field] = data[value_field].astype(np.float32)
    convert_2_daily_values(data, value_field)

def convert_2_daily_values(data, value_fiel):
    for row_index, row in data.iterrows():   
        days_in_month = pd.Period( str(row_index) ).days_in_month
        row[value_fiel] = row[value_fiel] / days_in_month

if __name__ == '__main__':
    milk = pd.read_csv(r'..\..\Data\monthly-milk-production.csv', ';', index_col=['month'], parse_dates=['month'], dayfirst=True)
    preprocess(milk, "milk")

    # 1.2.1. Season diff
    milk["milk"] = milk["milk"].diff(periods=12)
    # 1.2.2. diff
    milk["milk"] = milk["milk"].diff()
    #print(milk["milk"].iloc[:30])
    milk = milk.iloc[13:, :]

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html
    plot_acf(milk["milk"], lags=50)
    plt.grid(True)
    plt.show()
    
    from statsmodels.tsa.arima_model import ARIMA

    #model = ARIMA(milk["milk"], order=(4,1,2))
    model = ARIMA(milk["milk"], order=(0,0,2))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    print(model_fit.resid.sum())

    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()