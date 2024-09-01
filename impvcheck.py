import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import csv
import os
import requests

# Function to fetch and display the current Bitcoin price with a timestamp
def check_btc_price():
    # Fetch the latest Bitcoin price data from Yahoo Finance
    btc_data = yf.download('BTC-USD', period='1d', interval='1m')  # Using 1-minute interval to get the latest data
    current_price = btc_data['Close'].iloc[-1]  # Get the most recent closing price

    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Output the current price and timestamp
    print(f"Timestamp: {timestamp}")
    print(f"Current Bitcoin price: {current_price:.2f} USD")

    return current_price

# Function to check the predicted price from 24 hours ago in the log file and calculate accuracy
def check_prediction_24_hours_ago(log_file, accuracy_log_file):
    # Load the log file into a pandas DataFrame
    df = pd.read_csv(log_file)

    # Convert 'Timestamp' column to datetime format for accurate comparison
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Calculate the target time window for 24 hours ago
    target_time_24_hours_ago = datetime.now() - timedelta(days=1)
    time_window_start = target_time_24_hours_ago - timedelta(hours=1)  # 1 hour before
    time_window_end = target_time_24_hours_ago + timedelta(hours=1)    # 1 hour after

    # Filter for entries within the time window
    filtered_df = df[(df['Timestamp'] >= time_window_start) & (df['Timestamp'] <= time_window_end)]

    if filtered_df.empty:
        print(f"No prediction found within 1 hour of 24 hours ago: {target_time_24_hours_ago}.")
        return None  # Return None if no prediction is found

    # Find the entry closest to the target time within the filtered data
    closest_row = filtered_df.iloc[(filtered_df['Timestamp'] - target_time_24_hours_ago).abs().argsort()[:1]]

    # Fetch the predicted price from the closest timestamp
    closest_timestamp = closest_row['Timestamp'].values[0]
    predicted_price_24_hours_ago = closest_row['Predicted Price'].values[0]

    # Fetch the current Bitcoin price
    current_price = check_btc_price()

    # Calculate the percentage accuracy
    percentage_accuracy = ((current_price - predicted_price_24_hours_ago) / predicted_price_24_hours_ago) * 100

    # Output the results
    print(f"Timestamp 24 Hours Ago (Closest): {closest_timestamp}")
    print(f"Predicted Bitcoin Price 24 Hours Ago: {predicted_price_24_hours_ago:.2f} USD")
    print(f"Current Bitcoin Price: {current_price:.2f} USD")
    print(f"Percentage Difference: {percentage_accuracy:.2f}%")

    # Log the percentage difference to a separate accuracy CSV file
    header = ['Current Timestamp', 'Predicted Timestamp', 'Predicted Price', 'Actual Price', 'Percentage Difference']

    # Check if the accuracy log file exists and add the header if not
    file_exists = os.path.isfile(accuracy_log_file)
    with open(accuracy_log_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists or os.path.getsize(accuracy_log_file) == 0:
            writer.writerow(header)  # Write the header if the file doesn't exist or is empty
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), closest_timestamp,
                         predicted_price_24_hours_ago, current_price, percentage_accuracy])

    return percentage_accuracy  # Return percentage accuracy for notification

# Function to send ntfy notification with dynamic content
def send_ntfy_notification(percentage_accuracy):
    topic = 'btc-script-run'  # Replace with your chosen topic
    message = f'impvcheck.py script completed successfully. Percentage Accuracy: {percentage_accuracy:.2f}%'
    url = f'https://ntfy.sh/{topic}'

    # Send the notification as an HTTP POST request
    response = requests.post(url, data=message)

    if response.status_code == 200:
        print('Notification sent successfully!')
    else:
        print(f'Failed to send notification. Status code: {response.status_code}')

# Example usage: Check prediction made 24 hours ago
prediction_log_file = '/home/t0fum4n/BTCML/btc_price_predictions.csv'  # Path to the log file with predictions
accuracy_log_file = '/home/t0fum4n/BTCML/btc_price_accuracy_check.csv'  # Path to the separate log file for accuracy checks

# Calculate percentage accuracy and send notification
percentage_accuracy = check_prediction_24_hours_ago(prediction_log_file, accuracy_log_file)

# Send notification only if percentage_accuracy is calculated
if percentage_accuracy is not None:
    send_ntfy_notification(percentage_accuracy)
