import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st 


import datetime
st.title('stock price prediction')
start = '2015-03-01'
end = datetime.datetime.now().strftime('%Y-%m-%d')
stock = 'AAPL'

df = yf.download(stock, start=start, end=end)
df.head()