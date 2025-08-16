import pandas as pd
from datetime import datetime

# Function to load and preprocess the Bitstamp data by reading the CSV file,
# converting the 'Timestamp' column to datetime, and setting it as the index.
def dataload():
    bitstamp = pd.read_csv("../../data/btcusd_1-min_data.csv")
    bitstamp['Timestamp'] = pd.to_datetime(bitstamp['Timestamp'], unit='s')
    bitstamp.set_index('Timestamp', inplace=True)
    return bitstamp

def data_filter(bitstamp, start_date, end_date):
    return bitstamp[(bitstamp.index >= start_date) & (bitstamp.index <= end_date)]

# Function to add missing timestamps to the DataFrame based on a specified frequency.
# It creates a full date range from the minimum to maximum timestamp in the DataFrame,
def add_missing_timestamps(df: pd.DataFrame, freq='1min'):
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df_full = df.reindex(full_index)
    df_full.index.name = 'Timestamp'
    return df_full

# Function to fill missing values in the DataFrame using forward fill method.
# This method propagates the last valid observation forward to fill gaps.
def fill_missing_values(df: pd.DataFrame, method='ffill') -> pd.DataFrame:
    return df.fillna(method=method)
