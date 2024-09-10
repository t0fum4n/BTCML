import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime
import csv

# Load the data
data = pd.read_csv('btc_price_accuracy_check.csv')

# Select features for training (timestamp is excluded)
features = ['Predicted Price', 'Actual Price']

# Preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

# Define look-back period
look_back = 90


# Prepare the dataset for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 1])  # Actual price as the target
    return np.array(X), np.array(Y)


X, y = create_dataset(scaled_data, look_back)

# Reshape the data to be [samples, time steps, features] as required for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X, y, epochs=20, batch_size=1, verbose=2)

# Prepare test data (most recent 24 hours for prediction)
test_data = scaled_data[-look_back:, 0].reshape(1, look_back, 1)

# Make prediction for the next time step (24 hours from now)
predicted_price_scaled = model.predict(test_data)
predicted_price = scaler.inverse_transform(np.concatenate((predicted_price_scaled, np.zeros((1, 1))), axis=1))[0][0]

# Calculate performance metrics (Mean Absolute Error, Mean Squared Error, RMSE, etc.)
test_predict_prices = model.predict(X)
test_predict_prices = scaler.inverse_transform(
    np.concatenate((test_predict_prices, scaled_data[look_back:, 1:]), axis=1))[:, 0]
actual_prices = scaler.inverse_transform(scaled_data[look_back:, :])[:, 1]

# Metrics calculations
mae = mean_absolute_error(actual_prices, test_predict_prices)
mse = mean_squared_error(actual_prices, test_predict_prices)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((actual_prices - test_predict_prices) / actual_prices)) * 100
accuracy_score = 100 - mape

# Current price from the most recent actual data
current_price = data['Actual Price'].iloc[-1]

# Percentage change between current and predicted prices
percentage_change = ((predicted_price - current_price) / current_price) * 100

print(f"Predicted Price for 24 hours from now: {predicted_price}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Percentage Error: {mape}%")
print(f"Accuracy Score: {accuracy_score}%")

# Logging prediction and metrics to a separate CSV file
log_filename = 'btc_price_prediction_log.csv'
log_data = [
    {
        'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Current Price': current_price,
        'Predicted Price': predicted_price,
        'Percentage Change': percentage_change,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'Accuracy Score': accuracy_score
    }
]

# Write or append to log file
with open(log_filename, mode='a', newline='') as file:
    writer = csv.DictWriter(file,
                            fieldnames=['Timestamp', 'Current Price', 'Predicted Price', 'Percentage Change', 'MAE',
                                        'MSE', 'RMSE', 'MAPE', 'Accuracy Score'])

    # If the file is empty, write the header
    if file.tell() == 0:
        writer.writeheader()

    # Write the log data
    writer.writerows(log_data)

print(f"Prediction logged in {log_filename}")
