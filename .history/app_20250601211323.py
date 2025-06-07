import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import datetime

# Custom CSS for styling
st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
    
    .sidebar-content {
        padding: 1rem;
    }
    .profile-header, .project-header {
        color: #2c3e50;
        margin-bottom: 1rem;
        margin-top: 2rem;
    }
    .profile-text, .project-text {
        color: white;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    .project-features {
        margin-top: 1rem;
        padding-left: 1.2rem;
    }
    .project-features li {
        margin-bottom: 0.5rem;
        color: white;
    }
    .social-links {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 15px;
        margin-top: 1rem;
    }
    .social-icon {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        text-decoration: none !important;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        background: #000;
        border: 2px solid #000;
    }
    .social-icon i {
        font-size: 24px;
        color: white;
        transition: all 0.3s ease;
        z-index: 2;
    }
    .social-icon:hover {
        transform: translateY(-5px) rotate(5deg);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        background: white;
    }
    .social-icon:hover i {
        transform: scale(1.2);
        color: #000;
    }
    .divider {
        height: 1px;
        background: #e0e0e0;
        margin: 2rem 0;
    }
    </style>
    
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    with st.expander("👤 About Me", expanded=True):
        st.markdown("""
            <div class='profile-text'>
            Hi, I'm <strong>Kaithala Akhiranandan</strong>, a second-year B.Tech student specializing in Artificial Intelligence 
            and Machine Learning at ACE Engineering College. I have a passion for technology, and my skills span across 
            programming languages like Java and Python, as well as frontend development. I enjoy creating dynamic and 
            visually appealing user interfaces, and I'm always keen on learning new technologies and improving my coding abilities.
            </div>
            
            <div class='social-links'>
                <a href='https://github.com/akhiranandan-04' target='_blank' class='social-icon'>
                    <i class="fab fa-github"></i>
                </a>
                <a href='https://www.linkedin.com/in/akhiranandan-kaithala-a46a86291/' target='_blank' class='social-icon'>
                    <i class="fab fa-linkedin-in"></i>
                </a>
                <a href='https://www.instagram.com/akhiranandan_04' target='_blank' class='social-icon'>
                    <i class="fab fa-instagram"></i>
                </a>
                <a href='mailto:akhiranandankaithala@gmail.com' class='social-icon'>
                    <i class="fas fa-envelope"></i>
                </a>
            </div>
        """, unsafe_allow_html=True)

    with st.expander("📊 About Project", expanded=True):
        st.markdown("""
            <div class='project-text'>
            This Stock Market Predictor is an advanced machine learning application that helps investors and traders make informed decisions. It combines historical data analysis with deep learning to predict future stock price movements.
            
            <ul class='project-features'>
                <li>Real-time stock data fetching using Yahoo Finance API</li>
                <li>Interactive visualizations with moving averages (100 & 200 days)</li>
                <li>Deep learning model trained on historical price patterns</li>
                <li>Price prediction visualization compared to actual trends</li>
                <li>Support for any publicly traded stock symbol</li>
            </ul>
            </div>
        """, unsafe_allow_html=True)

# Main content
st.header('Stock Price Predictor')
stock =st.text_input('Enter Stock Symbol', 'AAPL')
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

# Summary Section
st.subheader('Stock Summary Statistics')
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Price Statistics")
    current_price = df['Close'].iloc[-1]
    st.metric(
        label="Current Price",
        value = f"${current_price.iloc[0]:.2f}"

    )
    
    high_price = df['High'].max()
    low_price = df['Low'].min()
    st.metric(
        label="All-Time High",
        value = f"${high_price.iloc[0]:.2f}"

    )
    st.metric(
        label="All-Time Low",
    value=f"${low_price.iloc[0]:.2f}"

    )

with col2:
    st.markdown("### Volume Statistics")
    latest_volume = df['Volume'].iloc[-1]
    avg_volume = df['Volume'].mean()
    
    st.metric(
        label="Latest Volume",
        value=f"{latest_volume.iloc[0]:,.0f}"

    )
    st.metric(
        label="Average Volume",
        value=f"{avg_volume:,.0f}"
    )

# Volume Chart
st.subheader('Trading Volume History')
fig3 = plt.figure(figsize=(12,6))
plt.plot(df.index, df['Volume'], color='purple', alpha=0.6)
plt.fill_between(df.index, df['Volume'], color='purple', alpha=0.2)
plt.grid(True, alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Volume')
st.pyplot(fig3)
plt.close()

