import numpy as np
import pandas as pd

def plot(data, value_field, title=""):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 5), dpi=300)
    plt.plot( data.index, data[value_field], linewidth=1)
    plt.title(title)
    plt.grid(True)
    plt.show()
    return fig

def dfuller(series):
    import statsmodels.api as sm
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
    print(milk["milk"].iloc[:10])
    convert_2_daily_values(milk, "milk")
    print(milk["milk"].iloc[:10])
    print(milk["milk"].sum())
    plot(milk, "milk")

    
    
    