import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime
import time
import json

st.title('Stock Market Prediction')

# Load JSON data
@st.cache_data
def load_json_data():
    with open('./Stock_Data.json', 'r') as f:
        data = json.load(f)
    return data

data_json = load_json_data()

def Load_data_json(Symbol):
    """Load stock data for a given symbol"""
    stock = data_json[Symbol]

    # Create DataFrame
    dates = stock["Date"]
    opens = stock["Price History"]["open"]
    highs = stock["Price History"]["high"]
    lows = stock["Price History"]["low"]
    closes = stock["Price History"]["close"]
    volumes = stock["volume"]

    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Open': pd.to_numeric(opens, errors='coerce'),
        'High': pd.to_numeric(highs, errors='coerce'),
        'Low': pd.to_numeric(lows, errors='coerce'),
        'Close': pd.to_numeric(closes, errors='coerce'),
        'Volume': pd.to_numeric(volumes, errors='coerce')
    })

    df = df.sort_values('Date').reset_index(drop=True)
    return df

def InfoComp(Symbol):
    """Get company information"""
    stock = data_json[Symbol]
    stock_info = {
        'Name': stock.get('Name', 'N/A'),
        'Founder': stock.get('Founder', 'N/A'),
        'Date of Birth': stock.get('Date of Birth', 'N/A'),
        'Industry': stock.get('Industry', 'N/A'),
        'Market Value': stock.get('Market Value', 'N/A')
    }
    return stock_info

# Get list of available symbols
symbols = list(data_json.keys())

# Create the dropdown menu
st.subheader("Select Company Symbol")
selected_company = st.selectbox('', symbols)

# Progress bar
my_bar = st.progress(0)
data_load_state = st.text('Loading data...')

# Load data
data = Load_data_json(selected_company)

for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1)

if percent_complete == 99:
    st.success(f'Data Loaded !', icon="✅")
    my_bar.empty()
    data_load_state.empty()

# Display data
st.dataframe(data)

# Information about the company
st.subheader("Info About the company")
st.write(InfoComp(selected_company))

# Statistics
st.subheader("Calculating data statistics")
st.dataframe(data.describe())

# Prepare data for machine learning
st.subheader("Creating Training Data Features:")

# Create features (Open, High, Low) to predict Close
X = data[['Open', 'High', 'Low']].values
y = data['Close'].values
dates = data['Date'].values

# Split data into training and testing (75% train, 25% test)
split_point = int(len(data) * 0.75)
X_train = X[:split_point]
y_train = y[:split_point]
X_test = X[split_point:]
y_test = y[split_point:]
dates_train = dates[:split_point]
dates_test = dates[split_point:]

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Display prediction results
st.subheader(f"Model Performance for {selected_company}:")
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.4f}")
col2.metric("Mean Squared Error", f"{mse:.4f}")
col3.metric("Mean Absolute Error", f"{mae:.4f}")

# Plot historical stock prices
st.subheader(f"Stock Market History for {selected_company}:")
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(dates, y, label="Close Price", linewidth=2)
plt.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Dollar US")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Plot predictions
st.subheader(f"Stock Market {selected_company} Prediction:")
fig2, ax2 = plt.subplots(figsize=(16, 8))

# Plot actual prices
ax2.plot(dates, y, label="Actual Close Price", linewidth=2, color='blue')

# Plot predictions
ax2.plot(dates_test, y_pred, label="Predicted Price", linewidth=2, color='red', linestyle='--')

plt.legend()

# Date range selector
date_start = st.date_input("Start Date", datetime.date(2018, 1, 1))
date_end = st.date_input("End Date", datetime.date(2023, 12, 30))

import matplotlib.dates as mdates
ax2.set_xlim(mdates.date2num([date_start, date_end]))

ax2.set_xlabel("Date")
ax2.set_ylabel("Dollar US")
ax2.grid(True, alpha=0.3)

st.pyplot(fig2)

# Show prediction dataframe
st.subheader("Prediction Details:")
pred_df = pd.DataFrame({
    'Date': dates_test,
    'Actual Close': y_test,
    'Predicted Close': y_pred,
    'Difference': y_test - y_pred,
    'Error %': ((y_test - y_pred) / y_test * 100)
})
st.dataframe(pred_df)
