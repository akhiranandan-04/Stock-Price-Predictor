import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import datetime



# About Project Section
st.markdown("## About the Project")
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.write("""
This Stock Market Predictor is a sophisticated tool that leverages machine learning to forecast stock prices. 
Built using Python and popular libraries like Keras and yfinance, it provides valuable insights through 
interactive visualizations powered by Streamlit.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Features")
    st.markdown("""
    - Real-time stock data fetching
    - Interactive price charts
    - Moving average analysis
    - Machine learning predictions
    """)

with col2:
    st.markdown("### Technologies")
    st.markdown("""
    - Python
    - Keras/TensorFlow
    - Streamlit
    - Pandas & NumPy
    """)

with col3:
    st.markdown("### Benefits")
    st.markdown("""
    - Data-driven decisions
    - Visual analysis
    - Historical trends
    - Future predictions
    """)

st.markdown("</div>", unsafe_allow_html=True)

# Means Section
st.markdown("## What This Means For You")
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### For Investors")
    st.write("""
    Get data-driven insights to make informed investment decisions. Our tool provides clear visualizations 
    and predictions to help you understand market trends and potential opportunities.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### For Analysts")
    st.write("""
    Access sophisticated technical analysis tools and machine learning predictions to enhance your research 
    and analysis capabilities. Compare different indicators and validate your strategies.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

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


from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Portfolio Website",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    .social-links {
        display: flex;
        gap: 20px;
    }
    .social-links a {
        color: #666;
        text-decoration: none;
    }
    .social-links a:hover {
        color: #157DA0;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# Navigation
st.markdown("<h1 style='text-align: center;'>Stock Market Predictor</h1>", unsafe_allow_html=True)
nav = st.container()
with nav:
    cols = st.columns(3)
    with cols[0]:
        st.markdown("<div style='text-align: center;'><a href='#about'>About Me</a></div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown("<div style='text-align: center;'><a href='#project'>About Project</a></div>", unsafe_allow_html=True)
    with cols[2]:
        st.markdown("<div style='text-align: center;'><a href='#means'>Means</a></div>", unsafe_allow_html=True)

# About Me Section
st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])

with col1:
    image = Image.open("WhatsApp Image 2024-09-07 at 12.00.44 PM.jpeg")
    st.image(image, use_column_width=True)

with col2:
    st.markdown("## Kaithala Akhiranandan")
    st.write("""
    Hi, I'm Kaithala Akhiranandan, a second-year B.Tech student specializing in Artificial Intelligence 
    and Machine Learning at ACE Engineering College. I have a passion for technology, and my skills span 
    across programming languages like Java and Python, as well as frontend development. I enjoy creating 
    dynamic and visually appealing user interfaces, and I'm always keen on learning new technologies and 
    improving my coding abilities.
    """)
    
    st.write("""
    In addition to my technical skills, I'm also deeply interested in design and editing. Whether it's 
    working on graphic designs or video editing, I love using creativity to bring ideas to life. Beyond 
    my academic and creative pursuits, I'm a huge fan of watching movies and anime, which inspire me 
    with their storytelling and visuals.
    """)
    
    st.markdown("""
    <div class='social-links'>
        <a href="https://www.linkedin.com/in/akhiranandan-kaithala" target="_blank">LinkedIn</a>
        <a href="https://github.com/akhiranandan-04" target="_blank">GitHub</a>
        <a href="mailto:akhiranandankaithala@gmail.com">Email</a>
        <a href="https://www.instagram.com/akhiranandan_04" target="_blank">Instagram</a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# About Project Section
st.markdown("## About the Project")
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.write("""
This Stock Market Predictor is a sophisticated tool that leverages machine learning to forecast stock prices. 
Built using Python and popular libraries like Keras and yfinance, it provides valuable insights through 
interactive visualizations powered by Streamlit.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Features")
    st.markdown("""
    - Real-time stock data fetching
    - Interactive price charts
    - Moving average analysis
    - Machine learning predictions
    """)

with col2:
    st.markdown("### Technologies")
    st.markdown("""
    - Python
    - Keras/TensorFlow
    - Streamlit
    - Pandas & NumPy
    """)

with col3:
    st.markdown("### Benefits")
    st.markdown("""
    - Data-driven decisions
    - Visual analysis
    - Historical trends
    - Future predictions
    """)

st.markdown("</div>", unsafe_allow_html=True)