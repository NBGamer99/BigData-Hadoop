import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import json
import plotly.express as px
import plotly.graph_objects as go
import os

# Load data
@st.cache_data
def load_json_data():
    with open('./Stock_Data.json', 'r') as f:
        data = json.load(f)
    return data

data = load_json_data()

########################################################################
# Toolspack

COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

def convert_to_float(list):
    return [float(x) for x in list]

def get_Stocks(SYMBOL):
    STOCK = data[SYMBOL]
    PRICES = STOCK['Price History']

    DATA = [
        STOCK['Date'],
        PRICES['open'],
        PRICES['high'],
        PRICES['low'],
        PRICES['close'],
        STOCK['volume']
    ]
    df = pd.DataFrame(np.array(DATA).T, columns=COLUMNS)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df['Open'] = pd.to_numeric(df['Open'], downcast='float')
    df['High'] = pd.to_numeric(df['High'], downcast='float')
    df['Low'] = pd.to_numeric(df['Low'], downcast='float')
    df['Close'] = pd.to_numeric(df['Close'], downcast='float')
    df['Volume'] = pd.to_numeric(df['Volume'], downcast='float')
    return df

def get_infos(SYMBOL):
    STOCK = data[SYMBOL]
    INFOS = {}
    INFOS["SYMBOL"] = SYMBOL
    INFOS["NAME"] = STOCK["Name"]
    INFOS["Description"] = STOCK["Description"]
    INFOS["Founder"] = STOCK["Founder"]
    INFOS["Country"] = STOCK["Country"]
    INFOS["Foundation Date"] = STOCK["Date of Birth"]
    INFOS["Industry"] = STOCK["Industry"]
    INFOS["Market Value"] = STOCK["Market Value"]
    return INFOS

def get_Symbols(data):
    return (tuple(data.keys()))

########################################################################

# Company Infos
COMPANIES = get_Symbols(data)

st.title("Stock Market Predictions with LSTM")
option = st.selectbox('Enter Stock Symbol', COMPANIES)

df = get_Stocks(str(option))

INFOS = get_infos(str(option))
st.write('Company Infos:', INFOS)

st.subheader('Data from 2010 to 2019')
st.write(df.describe())

st.subheader('Closing Price during time')
fig = px.line(df.Close, x=df.index, y=df.Close)
st.write(fig)

st.subheader('Candle Graph to visualize Open, High, Low, Close Price during time')
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
st.plotly_chart(fig)

st.subheader('Volume distribution during time')
fig = px.bar(df.Volume, x=df.index, y=df.Volume, color=df.Volume)
st.write(fig)

# Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# Loading the data
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Check if model exists, otherwise train a new one
model_path = './keras_model.h5'

if os.path.exists(model_path):
    # Load the existing model
    model = load_model(model_path)
    st.success('Pre-trained model loaded successfully!')
else:
    st.warning('Pre-trained model not found. Training a new model... This may take a few minutes.')

    # Building the Model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Show training progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Train with fewer epochs for demo purposes
    epochs = 20
    for epoch in range(epochs):
        model.fit(x_train, y_train, epochs=1, verbose=0)
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f'Training: Epoch {epoch + 1}/{epochs}')

    # Save the model
    model.save(model_path)
    st.success('Model trained and saved successfully!')
    progress_bar.empty()
    status_text.empty()

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

# Testing the model
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Making Predictions
y_predicted = model.predict(x_test)

# Normalize the data
scale_factor = 1/float(scaler.scale_[0])
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Calculate metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse = mean_squared_error(y_test, y_predicted)
mae = mean_absolute_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)

st.subheader('Model Performance Metrics:')
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.4f}")
col2.metric("Mean Squared Error", f"{mse:.4f}")
col3.metric("Mean Absolute Error", f"{mae:.4f}")

st.subheader('Model Prediction')
fig = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label="Original Price")
plt.plot(y_predicted, 'r', label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price in USD")
plt.legend()
plt.grid(True, alpha=0.3)
st.pyplot(fig)
