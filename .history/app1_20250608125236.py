import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler

# ------------------ Streamlit App: Stock Price Predictor ------------------ #


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
    # with st.expander("👤 About Me", expanded=True):
    #     st.markdown("""
    #         <div class='profile-text'>
    #         Hi, I'm <strong>Kaithala Akhiranandan</strong>, a second-year B.Tech student specializing in Artificial Intelligence 
    #         and Machine Learning at ACE Engineering College. I have a passion for technology, and my skills span across 
    #         programming languages like Java and Python, as well as frontend development. I enjoy creating dynamic and 
    #         visually appealing user interfaces, and I'm always keen on learning new technologies and improving my coding abilities.
    #         </div>
            
    #         <div class='social-links'>
    #             <a href='https://github.com/akhiranandan-04' target='_blank' class='social-icon'>
    #                 <i class="fab fa-github"></i>
    #             </a>
    #             <a href='https://www.linkedin.com/in/akhiranandan-kaithala-a46a86291/' target='_blank' class='social-icon'>
    #                 <i class="fab fa-linkedin-in"></i>
    #             </a>
    #             <a href='https://www.instagram.com/akhiranandan_04' target='_blank' class='social-icon'>
    #                 <i class="fab fa-instagram"></i>
    #             </a>
    #             <a href='mailto:akhiranandankaithala@gmail.com' class='social-icon'>
    #                 <i class="fas fa-envelope"></i>
    #             </a>
    #         </div>
    #     """, unsafe_allow_html=True)

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

# ------------------ Main App ------------------ #

st.title('📈 Stock Price Predictor')

# Input: Stock symbol
ticker = st.text_input('Enter Stock Symbol (e.g. AAPL, GOOGL, TSLA)', 'AAPL')
start_date = '2015-03-01'
end_date = datetime.datetime.now().strftime('%Y-%m-%d')

# Fetch stock data from yfinance
try:
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error("No data found for this stock symbol.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching stock data: {e}")
    st.stop()

# Data Overview
st.subheader('Stock Data Overview')
st.write(df.describe())

# Plot closing price
def plot_chart(title, *args):
    fig = plt.figure(figsize=(12,6))
    for data in args:
        plt.plot(data)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig)

st.subheader('📉 Closing Price Over Time')
plot_chart('Closing Price', df['Close'])

# Moving Averages
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
st.subheader('📊 Closing Price with 100 & 200-Day Moving Averages')
plot_chart('Moving Averages', df['Close'], ma100, ma200)

# Split data into train/test
train_data = df['Close'][:int(len(df)*0.7)].to_frame()
test_data = df['Close'][int(len(df)*0.7):].to_frame()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
train_array = scaler.fit_transform(train_data)

# Load model
try:
    model = load_model('keras_model.keras')  # Use relative path on Render
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Prepare test data for prediction
past_100_days = train_data.tail(100)
final_df = pd.concat([past_100_days, test_data], ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

# Reverse scaling
y_predicted = y_predicted * (1 / scaler.scale_[0])
y_test = y_test * (1 / scaler.scale_[0])

# Plot prediction vs original
st.subheader('🔮 Predicted vs Actual Closing Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'g', label='Actual Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig2)

# Metrics
st.subheader('📌 Stock Summary Statistics')
col1, col2 = st.columns(2)

with col1:
    st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
    st.metric("All-Time High", f"${df['High'].max():.2f}")
    st.metric("All-Time Low", f"${df['Low'].min():.2f}")

with col2:
    st.metric("Latest Volume", f"{int(df['Volume'].iloc[-1]):,}")
    st.metric("Average Volume", f"{int(df['Volume'].mean()):,}")

# Volume chart
st.subheader('📦 Trading Volume History')
fig3 = plt.figure(figsize=(12,6))
plt.plot(df.index, df['Volume'], color='purple', alpha=0.6)
plt.fill_between(df.index, df['Volume'], color='purple', alpha=0.2)
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True)
st.pyplot(fig3)
plt.close()