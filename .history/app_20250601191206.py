import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import datetime

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

import React from 'react';
import { Github, Linkedin, Mail, Instagram } from 'lucide-react';
src="/WhatsApp Image 2024-09-07 at 12.00.44 PM.jpeg"

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="ibg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <span className="text-2xl font-bold text-gray-800">Stock Market Predictor</span>
            <div className="flex space-x-8">
              <a href="#about" className="text-gray-600 hover:text-gray-900">About Me</a>
              <a href="#project" className="text-gray-600 hover:text-gray-900">About Project</a>
              <a href="#means" className="text-gray-600 hover:text-gray-900">Means</a>
            </div>
          </div>
        </div>
      </nav>

      {/* About Me Section */}
      <section id="about" className="py-20 bg-white">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
            <div className="md:flex">
              <div className="md:w-1/3 p-8">
                <img
                  src="/WhatsApp Image 2024-09-07 at 12.00.44 PM.jpeg"
                  alt="Profile"
                  className="rounded-lg w-full h-auto object-cover"
                />
              </div>
              <div className="md:w-2/3 p-8">
                <h2 className="text-3xl font-bold text-gray-800 mb-4">Kaithala Akhiranandan</h2>
                <p className="text-gray-600 mb-6">
                  Hi, I'm Kaithala Akhiranandan, a second-year B.Tech student specializing in Artificial Intelligence and Machine Learning at ACE Engineering College. I have a passion for technology, and my skills span across programming languages like Java and Python, as well as frontend development. I enjoy creating dynamic and visually appealing user interfaces, and I'm always keen on learning new technologies and improving my coding abilities.
                </p>
                <p className="text-gray-600 mb-6">
                  In addition to my technical skills, I'm also deeply interested in design and editing. Whether it's working on graphic designs or video editing, I love using creativity to bring ideas to life. Beyond my academic and creative pursuits, I'm a huge fan of watching movies and anime, which inspire me with their storytelling and visuals.
                </p>
                <div className="flex space-x-4">
                  <a href="https://www.linkedin.com/in/akhiranandan-kaithala" className="text-gray-600 hover:text-blue-600">
                    <Linkedin className="w-6 h-6" />
                  </a>
                  <a href="https://github.com/akhiranandan-04" className="text-gray-600 hover:text-gray-900">
                    <Github className="w-6 h-6" />
                  </a>
                  <a href="mailto:akhiranandankaithala@gmail.com" className="text-gray-600 hover:text-red-600">
                    <Mail className="w-6 h-6" />
                  </a>
                  <a href="https://www.instagram.com/akhiranandan_04" className="text-gray-600 hover:text-pink-600">
                    <Instagram className="w-6 h-6" />
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
