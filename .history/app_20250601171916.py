import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

import yfinance as yf
import datetime

st.title('Stock Price Prediction')

stock =st.text_input('Enter Stock Symnbol', 'AAPL')

start = '2015-03-01'
end = datetime.datetime.now().strftime('%Y-%m-%d')
stock = 'AAPL'

df = yf.download(stock, start=start, end=end)
df.head()

st.subheader('Data from 2015 - 2025')
st.write(df.describe())