import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error
from math import sqrt
import yfinance as yf
import threading

# Tunable Hyperparameters
TICKER = 'BTC-USD'
START_DATE = '2020-01-01'
END_DATE = None
LOOK_BACK = 120
TRAIN_TEST_SPLIT = 0.8

GRU_UNITS_LAYER_1 = 80
GRU_UNITS_LAYER_2 = 60
GRU_UNITS_LAYER_3 = 40
DROPOUT_RATE_LAYER_1 = 0.4
DROPOUT_RATE_LAYER_2 = 0.3
DROPOUT_RATE_LAYER_3 = 0.3
DENSE_UNITS = 20
L2_REGULARIZATION = 0.001

OPTIMIZER_LEARNING_RATE = 0.00005
BATCH_SIZE = 64
EPOCHS = 300
PATIENCE = 15

# Load historical Bitcoin price data from Yahoo Finance
def load_data_from_yahoo(ticker=TICKER, start_date=START_DATE, end_date=END_DATE):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close', 'Volume']]  # Reduced to only essential features

    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data.loc[:, 'RSI'] = rsi  # Using .loc to avoid SettingWithCopyWarning

    # Calculate MACD
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26

    # Create a composite feature combining MACD and RSI
    data.loc[:, 'Momentum'] = rsi * macd  # Composite feature to capture momentum

    # Additional feature: Bollinger Bands
    data.loc[:, 'MA20'] = data['Close'].rolling(window=20).mean()
    data.loc[:, '20dSTD'] = data['Close'].rolling(window=20).std()
    data.loc[:, 'UpperBand'] = data['MA20'] + (data['20dSTD'] * 2)
    data.loc[:, 'LowerBand'] = data['MA20'] - (data['20dSTD'] * 2)

    data.fillna(0, inplace=True)  # Handle NaN values from indicator calculations
    return data


# Preprocess the data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    train_size = int(len(scaled_data) * TRAIN_TEST_SPLIT)
    train_data = scaled_data[0:train_size]
    test_data = scaled_data[train_size:]

    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    # Dynamically determine the number of features
    n_features = X_train.shape[2] if len(X_train.shape) > 2 else 1

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], n_features)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_features)

    return X_train, y_train, X_test, y_test, scaler


# Create a dataset matrix for LSTM input
def create_dataset(data, look_back=LOOK_BACK):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back, 0])  # Targeting the "Close" price
    return np.array(X), np.array(y)


# Build the improved GRU model with further refinements
def build_improved_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(GRU(GRU_UNITS_LAYER_1, return_sequences=True,
                                kernel_regularizer=l2(L2_REGULARIZATION)),
                            input_shape=input_shape))  # Using GRU with increased units
    model.add(Dropout(DROPOUT_RATE_LAYER_1))  # Dropout rate at 0.4
    model.add(Bidirectional(GRU(GRU_UNITS_LAYER_2, return_sequences=True,
                                kernel_regularizer=l2(L2_REGULARIZATION))))  # Adding another GRU layer
    model.add(Dropout(DROPOUT_RATE_LAYER_2))
    model.add(Bidirectional(GRU(GRU_UNITS_LAYER_3, return_sequences=False,
                                kernel_regularizer=l2(L2_REGULARIZATION))))  # Final GRU layer
    model.add(Dropout(DROPOUT_RATE_LAYER_3))
    model.add(Dense(DENSE_UNITS, kernel_regularizer=l2(L2_REGULARIZATION)))
    model.add(Dense(1))

    # Use Nadam optimizer with a fine-tuned learning rate
    optimizer = Nadam(learning_rate=OPTIMIZER_LEARNING_RATE)  # Lowered learning rate for finer convergence
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


# Plot training history without blocking the main thread
def plot_history(history):
    def show_plot():
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()

    thread = threading.Thread(target=show_plot)
    thread.start()


# Main function to run the script
def main():
    # Load data from Yahoo Finance
    data = load_data_from_yahoo(start_date=START_DATE)

    # Preprocess data
    X_train, y_train, X_test, y_test, scaler = preprocess_data(data)

    # Build and train the improved model
    model = build_improved_model((X_train.shape[1], X_train.shape[2]))  # Updated to reflect actual features
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])

    # Plot training history
    plot_history(history)

    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Inverse transform predictions and actual values
    train_predictions = scaler.inverse_transform(
        np.concatenate((train_predictions, np.zeros((train_predictions.shape[0], X_train.shape[2] - 1))), axis=1))[:, 0]
    y_train = scaler.inverse_transform(
        np.concatenate((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], X_train.shape[2] - 1))), axis=1))[:, 0]
    test_predictions = scaler.inverse_transform(
        np.concatenate((test_predictions, np.zeros((test_predictions.shape[0], X_train.shape[2] - 1))), axis=1))[:, 0]
    y_test = scaler.inverse_transform(
        np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], X_train.shape[2] - 1))), axis=1))[:, 0]

    # Calculate RMSE
    train_rmse = sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = sqrt(mean_squared_error(y_test, test_predictions))
    print(f'Train RMSE: {train_rmse:.2f}')
    print(f'Test RMSE: {test_rmse:.2f}')

    # Predict the next day's price
    last_data = data[-LOOK_BACK:].values
    last_data_scaled = scaler.transform(last_data)
    last_data_scaled = last_data_scaled.reshape(1, -1, X_train.shape[2])  # Updated to reflect actual features
    next_day_prediction = model.predict(last_data_scaled)
    next_day_price = scaler.inverse_transform(
        np.concatenate((next_day_prediction, np.zeros((next_day_prediction.shape[0], X_train.shape[2] - 1))), axis=1))[
                     :, 0]

    print(f'Predicted Bitcoin price for the next day: ${next_day_price[0]:.2f}')


if __name__ == "__main__":
    main()
