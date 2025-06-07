import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import datetime

model = load_model(r'C:\Users\kaith\Downloads\stock_price_prediction\keras_model.keras')


st.header('Stock Market Predictor')

stock =st.text_input('Enter Stock Symnbol', 'AAPL')
start = '2015-03-01'
end = datetime.datetime.now().strftime('%Y-%m-%d')


df = yf.download(stock, start ,end)
df.head()
st.subheader('Stock Data from 2015 - 2025')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df["Close"])
st.pyplot(fig)