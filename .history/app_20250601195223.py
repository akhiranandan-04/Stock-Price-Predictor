import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import datetime

st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide",
)

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

st.subheader('Closing Price vs Time Chart with 100 days MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r', label='100-day MA')
plt.plot(df['Close'], color="#157DA0", label='Closing Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 & 200 days MA')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r', label='100-day MA')
plt.plot(ma200, 'g', label='200-day MA')
plt.plot(df['Close'], color="#157DA0", label='Closing Price')
plt.legend()
st.pyplot(fig)

#spliting data into training and testing

data_train = pd.DataFrame(df.Close[0: int(len(df)*0.70)])
data_test = pd.DataFrame(df.Close[int(len(df)*0.70): len(df)])
print(data_train.shape)
print(data_test.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_train_array = scaler.fit_transform(data_train)

#loadind model
model = load_model(r'C:\Users\kaith\Downloads\stock_price_prediction\keras_model.keras')

pas_100_days = data_train.tail(100)
final_df = pd.concat([pas_100_days, data_test], ignore_index=True)


x = []
y = []

for i in range(100, data_train_array.shape[0]):
    x.append(data_train_array[i-100:i])
    y.append(data_train_array[i,0])
x, y = np.array(x), np.array(y)
y_predicted=model.predict(x)

scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y = y * scale_factor


#final graph

st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y, 'g', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

