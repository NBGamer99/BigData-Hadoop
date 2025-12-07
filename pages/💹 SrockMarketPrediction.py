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

# Prepare data for machine learning - PROPER TIME SERIES PREDICTION
st.subheader("Creating Training Data Features:")

st.info("""
**📊 Prediction Strategy**: Using past 5 days of closing prices to predict the next day's closing price.
This is realistic because we only use historical data available at the time of prediction.
""")

# Create lagged features - use past data to predict future
def create_lagged_features(df, n_lags=5):
    """Create features using past n days to predict next day"""
    df_lagged = df.copy()

    # Create lagged features (previous days' prices)
    for i in range(1, n_lags + 1):
        df_lagged[f'Close_Lag{i}'] = df_lagged['Close'].shift(i)
        df_lagged[f'Volume_Lag{i}'] = df_lagged['Volume'].shift(i)

    # Create technical indicators
    df_lagged['Price_Change'] = df_lagged['Close'].pct_change()
    df_lagged['Moving_Avg_5'] = df_lagged['Close'].rolling(window=5).mean()
    df_lagged['Moving_Avg_20'] = df_lagged['Close'].rolling(window=20).mean()
    df_lagged['Volatility'] = df_lagged['Close'].rolling(window=5).std()

    # Target: Next day's close price
    df_lagged['Target'] = df_lagged['Close'].shift(-1)

    # Drop rows with NaN values
    df_lagged = df_lagged.dropna()

    return df_lagged

# Create lagged dataset
data_lagged = create_lagged_features(data, n_lags=5)

# Show sample of features
st.write("**Sample of engineered features:**")
feature_cols = [col for col in data_lagged.columns if 'Lag' in col or 'Moving_Avg' in col or 'Volatility' in col]
st.dataframe(data_lagged[['Date'] + feature_cols[:5] + ['Target']].head(10))

# Prepare features and target
feature_columns = [col for col in data_lagged.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
X = data_lagged[feature_columns].values
y = data_lagged['Target'].values
dates = data_lagged['Date'].values

# Split data into training and testing (75% train, 25% test)
split_point = int(len(X) * 0.75)
X_train = X[:split_point]
y_train = y[:split_point]
X_test = X[split_point:]
y_test = y[split_point:]
dates_train = dates[:split_point]
dates_test = dates[split_point:]

st.write(f"**Training samples:** {len(X_train)} | **Testing samples:** {len(X_test)}")

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
y_pred_train = regressor.predict(X_train)
y_pred = regressor.predict(X_test)

# Show feature importance
st.write("**Top 5 Most Important Features:**")
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': np.abs(regressor.coef_)
}).sort_values('Coefficient', ascending=False).head(5)
st.dataframe(feature_importance)

# Display prediction results
st.subheader(f"Model Performance for {selected_company}:")
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Calculate metrics for test set
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Calculate average price for context
avg_price = np.mean(y_test)

col1, col2, col3, col4 = st.columns(4)
col1.metric("R² Score", f"{r2:.4f}")
col2.metric("MAE", f"${mae:.2f}")
col3.metric("MAPE", f"{mape:.2f}%")
col4.metric("Avg Price", f"${avg_price:.2f}")

st.write(f"""
**Interpretation:**
- R² Score ({r2:.4f}): Explains {r2*100:.1f}% of price variance (realistic for stock prediction)
- MAE ({mae:.2f}): Average prediction error is {mae:.2f} per share
- MAPE ({mape:.2f}%): Average error is {mape:.2f}% of actual price
""")

if r2 > 0.95:
    st.warning("⚠️ R² > 0.95 might indicate data leakage. Stock markets are inherently unpredictable!")
elif r2 > 0.5:
    st.success("✅ Good model performance for stock prediction!")
else:
    st.info("ℹ️ Stock prices are difficult to predict. This is normal for financial data.")

# Plot historical stock prices
st.subheader(f"Stock Market History for {selected_company}:")
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(data['Date'].values, data['Close'].values, label="Historical Close Price", linewidth=2, color='blue')
plt.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Plot predictions vs actual
st.subheader(f"Stock Market {selected_company} - Prediction vs Actual:")

# Create figure with training and testing regions
fig2, ax2 = plt.subplots(figsize=(16, 8))

# Plot training data
ax2.plot(dates_train, y_train, label="Training Data (Actual)", linewidth=2, color='blue', alpha=0.7)

# Plot test actual prices
ax2.plot(dates_test, y_test, label="Test Data (Actual)", linewidth=2, color='green')

# Plot predictions (only for test period)
ax2.plot(dates_test, y_pred, label="Predicted Prices", linewidth=2, color='red', linestyle='--')

# Add vertical line to separate train/test
if len(dates_train) > 0:
    ax2.axvline(x=dates_train[-1], color='gray', linestyle=':', linewidth=2, label='Train/Test Split')

plt.legend(loc='best')

# Date range selector
col1, col2 = st.columns(2)
with col1:
    date_start = st.date_input("Start Date", datetime.date(2018, 1, 1))
with col2:
    date_end = st.date_input("End Date", datetime.date(2023, 12, 30))

import matplotlib.dates as mdates
ax2.set_xlim(mdates.date2num([date_start, date_end]))

ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.set_title(f"{selected_company} - Next-Day Price Predictions")
ax2.grid(True, alpha=0.3)

st.pyplot(fig2)

# Prediction accuracy over time
st.subheader("Prediction Error Over Time:")
fig3, ax3 = plt.subplots(figsize=(16, 6))

prediction_errors = y_test - y_pred
ax3.plot(dates_test, prediction_errors, label="Prediction Error", linewidth=1.5, color='purple')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.fill_between(dates_test, prediction_errors, 0, alpha=0.3, color='purple')

ax3.set_xlabel("Date")
ax3.set_ylabel("Error (USD)")
ax3.set_title("Prediction Error = Actual - Predicted")
ax3.grid(True, alpha=0.3)
plt.legend()

st.pyplot(fig3)

# Show prediction dataframe
st.subheader("Prediction Details (Test Period):")
pred_df = pd.DataFrame({
    'Date': dates_test,
    'Actual Close': y_test,
    'Predicted Close': y_pred,
    'Difference ($)': y_test - y_pred,
    'Error %': np.abs((y_test - y_pred) / y_test * 100),
    'Correct Direction': ['✅' if (i > 0 and (y_test[idx] > y_test[max(0, idx-1)])) or (i < 0 and (y_test[idx] < y_test[max(0, idx-1)])) else '❌'
                          for idx, i in enumerate(y_pred - y_test)]
})

# Format the dataframe
pred_df_display = pred_df.copy()
pred_df_display['Actual Close'] = pred_df_display['Actual Close'].apply(lambda x: f'${x:.2f}')
pred_df_display['Predicted Close'] = pred_df_display['Predicted Close'].apply(lambda x: f'${x:.2f}')
pred_df_display['Difference ($)'] = pred_df_display['Difference ($)'].apply(lambda x: f'${x:+.2f}')
pred_df_display['Error %'] = pred_df_display['Error %'].apply(lambda x: f'{x:.2f}%')

st.dataframe(pred_df_display.tail(20))

# Summary statistics
st.subheader("Prediction Summary:")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Best Prediction Error", f"${np.min(np.abs(y_test - y_pred)):.2f}")
with col2:
    st.metric("Worst Prediction Error", f"${np.max(np.abs(y_test - y_pred)):.2f}")
with col3:
    st.metric("Median Error", f"${np.median(np.abs(y_test - y_pred)):.2f}")

st.info("""
**📌 Note**: This model predicts the **next day's closing price** using the past 5 days of data.
Stock prices are inherently difficult to predict due to market volatility, news events, and other factors.
This model is for educational purposes and should not be used for actual trading decisions.
""")
