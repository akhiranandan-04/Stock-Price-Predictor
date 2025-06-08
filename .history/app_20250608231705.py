import numpy as np
import pandas as pd
from alpha_vantage.async_support.timeseries import TimeSeries
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
import asyncio
import os
from os import environ

# ------------------ API Key Check ------------------ #
def check_api_key():
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        st.error("""
        ⚠️ Alpha Vantage API key not found! Please follow these steps:

        1. Get a free API key from: https://www.alphavantage.co/support/#api-key
        2. Open PowerShell as administrator
        3. Run this command (replace with your actual API key):
           ```
           [System.Environment]::SetEnvironmentVariable('ALPHA_VANTAGE_API_KEY', 'YOUR_API_KEY', 'User')
           ```
        4. Close and reopen PowerShell
        5. Restart this Streamlit app
        """)
        st.stop()
    return api_key

# Get API key
api_key = check_api_key()

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
                <li>Real-time stock data fetching using Alpha Vantage API</li>
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

async def get_stock_data(symbol):
    # Initialize Alpha Vantage API
    ts = TimeSeries(key=api_key, output_format='pandas')
    try:
        # Fetch stock data from Alpha Vantage
        data, meta_data = await ts.get_daily(symbol=symbol, outputsize='full')
        await ts.close()  # Close the session
        # Rename columns to match our previous format
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Sort index to ascending order
        return data.sort_index(ascending=True)
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Use asyncio to run the async function
df = asyncio.run(get_stock_data(ticker))

if df is None or df.empty:
    st.error("No data found for this stock symbol.")
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

# Plot 100-day Moving Average
st.subheader('📊 Closing Price with 100-Day Moving Average')
fig_ma100 = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Closing Price', alpha=0.8)
plt.plot(ma100, label='100-Day MA', color='red', linewidth=2)
plt.title('100-Day Moving Average vs Closing Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.ylim(0, 1550)  # Set y-axis range
plt.legend()
plt.grid(True, alpha=0.3)
st.pyplot(fig_ma100)

# Plot 200-day Moving Average
st.subheader('📊 Closing Price with 200-Day Moving Average')
fig_ma200 = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Closing Price', alpha=0.8)
plt.plot(ma200, label='200-Day MA', color='blue', linewidth=2)
plt.title('200-Day Moving Average vs Closing Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.ylim(0, 1550)  # Set y-axis range
plt.legend()
plt.grid(True, alpha=0.3)
st.pyplot(fig_ma200)

# Combined Moving Averages Plot
st.subheader('📊 Combined Moving Averages Analysis')
fig_combined = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Closing Price', alpha=0.6)
plt.plot(ma100, label='100-Day MA', color='red', linewidth=2)
plt.plot(ma200, label='200-Day MA', color='blue', linewidth=2)
plt.title('Combined Moving Averages Analysis')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.ylim(0, 1550)  # Set y-axis range
plt.legend()
plt.grid(True, alpha=0.3)
st.pyplot(fig_combined)

# Add Moving Average Crossover Analysis
st.subheader('Moving Average Crossover Analysis')
crossover_text = """
The Moving Average Crossover Analysis helps identify potential trading signals:
- When the 100-day MA crosses above the 200-day MA: Potential bullish signal (Golden Cross)
- When the 100-day MA crosses below the 200-day MA: Potential bearish signal (Death Cross)
"""
st.write(crossover_text)

# Calculate current position of MAs
current_ma100 = ma100.iloc[-1]
current_ma200 = ma200.iloc[-1]
current_price = df['Close'].iloc[-1]

# Display current values
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Price", f"${current_price:.2f}")
with col2:
    st.metric("100-Day MA", f"${current_ma100:.2f}")
with col3:
    st.metric("200-Day MA", f"${current_ma200:.2f}")

# Market Position Analysis
if current_ma100 > current_ma200:
    if current_price > current_ma100:
        st.success("Strong Uptrend: Price is above both moving averages, and 100-day MA is above 200-day MA")
    else:
        st.info("Potential Uptrend: 100-day MA is above 200-day MA, but price is below 100-day MA")
else:
    if current_price < current_ma100:
        st.error("Strong Downtrend: Price is below both moving averages, and 100-day MA is below 200-day MA")
    else:
        st.warning("Potential Downtrend: 100-day MA is below 200-day MA, but price is above 100-day MA")

# Split data into train/test
train_data = df['Close'][:int(len(df)*0.7)]
test_data = df['Close'][int(len(df)*0.7):]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
train_array = scaler.fit_transform(np.array(train_data).reshape(-1, 1))

# Load model
try:
    model = load_model('keras_model.keras')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Prepare test data for prediction
past_100_days = train_data.tail(100)
final_df = pd.concat([past_100_days, test_data], ignore_index=True)
input_data = scaler.transform(np.array(final_df).reshape(-1, 1))

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



# Summary Section
st.subheader('Stock Summary Statistics')
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Price Statistics")
    current_price = df['Close'].iloc[-1]
    st.metric(
        label="Current Price",
        value=f"${current_price:.2f}"
    )
    
    high_price = df['High'].max()
    low_price = df['Low'].min()
    st.metric(
        label="All-Time High",
        value=f"${high_price:.2f}"
    )
    st.metric(
        label="All-Time Low",
        value=f"${low_price:.2f}"
    )

with col2:
    st.markdown("### Volume Statistics")
    latest_volume = df['Volume'].iloc[-1]
    avg_volume = df['Volume'].mean()
    
    st.metric(
        label="Latest Volume",
        value=f"{latest_volume:,.0f}"
    )
    st.metric(
        label="Average Volume",
        value=f"{avg_volume:,.0f}"
    )

print(type(df['Volume']))
print(df['Volume'].shape)

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