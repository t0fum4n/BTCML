import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from datetime import datetime
import csv
import os

# Tunable Parameters
START_DATE = '2018-01-01'
END_DATE = '2024-08-30'

TIME_STEP = 10
TRAIN_SIZE_RATIO = 0.6
LSTM_UNITS = 55
DROPOUT_RATE = 0.2
EPOCHS = 45
BATCH_SIZE = 32

# Feature Toggle Dictionary
feature_toggle = {
    'Close': True,
    'Open': True,
    'High': True,
    'Low': True,
    'Volume': True,
    'Returns': True,
    'MA50': True,
    'MA200': True,
}

# Step 1: Load Bitcoin data from Yahoo Finance
df = yf.download('BTC-USD', start=START_DATE, end=END_DATE)

# Include additional features based on the toggle
if feature_toggle['Returns']:
    df['Returns'] = df['Close'].pct_change()

if feature_toggle['MA50']:
    df['MA50'] = df['Close'].rolling(window=50).mean()

if feature_toggle['MA200']:
    df['MA200'] = df['Close'].rolling(window=200).mean()

if feature_toggle['Open']:
    df['Open'] = df['Open']

if feature_toggle['High']:
    df['High'] = df['High']

if feature_toggle['Low']:
    df['Low'] = df['Low']

if feature_toggle['Volume']:
    df['Volume'] = df['Volume']

df.dropna(inplace=True)

# Ensure there is enough data for TIME_STEP
if len(df) <= TIME_STEP:
    raise ValueError("Not enough data to create sequences with the specified TIME_STEP. "
                     "Increase the amount of data or reduce the TIME_STEP.")

# Prepare data for LSTM model
features = [key for key, value in feature_toggle.items() if value]
data = df[features].values

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create training and test datasets
train_size = int(len(scaled_data) * TRAIN_SIZE_RATIO)
train_data = scaled_data[0:train_size]
test_data = scaled_data[train_size:]

# Function to create datasets in time series format
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(train_data, TIME_STEP)
X_test, y_test = create_dataset(test_data, TIME_STEP)

# Ensure X_train and X_test are not empty
if X_train.size == 0 or X_test.size == 0:
    raise ValueError("X_train or X_test is empty. Not enough data after creating sequences. "
                     "Consider adjusting the TIME_STEP or checking the data preparation steps.")

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Build the LSTM model
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=LSTM_UNITS, return_sequences=True))
model.add(Dropout(DROPOUT_RATE))
model.add(LSTM(units=LSTM_UNITS, return_sequences=False))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# Predict the next day's price
last_60_days = test_data[-TIME_STEP:]
X_next = last_60_days.reshape((1, last_60_days.shape[0], last_60_days.shape[1]))

predicted_price = model.predict(X_next)
predicted_price = scaler.inverse_transform(np.concatenate((predicted_price, np.zeros((1, len(features) - 1))), axis=1))[:, 0]

# Fetch current price of Bitcoin
current_price = yf.download('BTC-USD', period='1d', interval='1m')['Close'].iloc[-1]

# Calculate percentage change
percentage_change = ((predicted_price[0] - current_price) / current_price) * 100

# Calculate Evaluation Metrics for Logging
test_predict = model.predict(X_test)
test_predict_prices = scaler.inverse_transform(
    np.concatenate((test_predict, np.zeros((test_predict.shape[0], len(features) - 1))), axis=1)
)[:, 0]
actual_prices = scaler.inverse_transform(
    np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], len(features) - 1))), axis=1)
)[:, 0]

mae = mean_absolute_error(actual_prices, test_predict_prices)
mse = mean_squared_error(actual_prices, test_predict_prices)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(actual_prices, test_predict_prices)

# Normalize metrics to a common scale (0 to 1)
normalized_mae = mae / np.max(actual_prices)
normalized_rmse = rmse / np.max(actual_prices)
normalized_mape = mape / 100  # Since MAPE is already a percentage

# Calculate composite score (average of normalized metrics)
composite_score = (normalized_mae + normalized_rmse + normalized_mape) / 3
accuracy_score = (1 - composite_score) * 100

# Timestamp for Logging
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Log the results to a CSV file
log_file = 'btc_price_predictions.csv'
header = ['Timestamp', 'Current Price', 'Predicted Price', 'Percentage Change', 'MAE', 'MSE', 'RMSE', 'MAPE', 'Accuracy Score']

# Check if the log file exists and add the header if not
file_exists = os.path.isfile(log_file)
with open(log_file, 'a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(header)  # Write the header if the file doesn't exist
    writer.writerow([timestamp, current_price, predicted_price[0], percentage_change, mae, mse, rmse, mape, accuracy_score])

# Output the logging information
print(f"Timestamp: {timestamp}")
print(f"Current Bitcoin price: {current_price:.2f} USD")
print(f"Predicted Bitcoin price for tomorrow: {predicted_price[0]:.2f} USD")
print(f"Percentage Change: {percentage_change:.2f}%")
print(f"Mean Absolute Error (MAE): {mae:.2f} USD")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} USD")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")
print(f"Model Accuracy Score: {accuracy_score:.2f}%")
