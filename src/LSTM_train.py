# =============================================================
# DEPRECATED: LSTM_train.py
# This script is no longer maintained. 
# All functionality has been migrated to notebooks to support execution in the Colab environment.
# =============================================================
from xml.parsers.expat import model
from matplotlib import units
import numpy as np
import os
import json
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import load_model, Sequential
from keras.layers import Layer, LSTM, Dense, Dropout
import keras.backend as K


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)  
        a = K.softmax(e, axis=1)               
        output = x * a                        
        return K.sum(output, axis=1)

# create sequences for LSTM
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size][0])  
    return np.array(X), np.array(y)

# create sequences for multi-step LSTM
def make_multistep_sequences(data, window_size=30, future_steps=5):

    X, y = [], []
    for i in range(len(data) - window_size - future_steps + 1):
        X.append(data[i : i + window_size, :])
        y.append(data[i + window_size : i + window_size + future_steps, 0])  
    return np.array(X), np.array(y)

def train_and_save_model(X_train, y_train, 
                         X_val, y_val,
                         model_index=0, 
                         layers=2, 
                         units=64, 
                         dropout_rate=0.2, 
                         batch_size=32, 
                         epochs=20,
                         window_size=24,
                         future_steps=1,
                         optimizer='adam',
                         callback=None,
                         result_file=None):

    # Build model
    model = Sequential()
    model.add(LSTM(units, return_sequences=(layers > 1), input_shape=(window_size, X_train.shape[2])))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    for i in range(1, layers):
        is_last = (i == layers - 1)
        model.add(LSTM(units, return_sequences=not is_last))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(Dense(future_steps))
    model.compile(optimizer=optimizer, loss='mae')

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callback
    )

    # Generate file names
    model_name = f"model_{model_index}_layers{layers}_units{units}.h5"
    history_name = f"history_{model_index}.json"
    model_path = os.path.join("models", model_name)
    history_path = os.path.join("histories", history_name)

    # Ensure folders exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("histories", exist_ok=True)

    try:
        # Save model
        model.save(model_path)

        # Save training history
        with open(history_path, "w") as f:
            json.dump(history.history, f)

        # Get best validation loss
        min_val_loss = min(history.history["val_loss"])

        # Write result to CSV
        # result_file = "training_results.csv"
        write_header = not os.path.exists(result_file)

        with open(result_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow([
                    "index","lstm_units", "num_layers", "dropout", "batch_size", "epochs",
                    "optimizer", "model_name", "val_loss", "history_path"
                ])
            writer.writerow([
                model_index,units, layers, dropout_rate, batch_size,
                epochs,  optimizer, model_name, min_val_loss, history_name
            ])

    except Exception as e:
        print("Error during save or logging:", e)

    return model, history

def train_and_save_model_with_attention(X_train, y_train, 
                         X_val, y_val,
                         model_index=0, 
                         layers=2, 
                         units=64, 
                         dropout_rate=0.2, 
                         batch_size=32, 
                         epochs=20,
                         window_size=24,
                         callback=None,
                         result_file=None):

    # Build model
    model = Sequential()

    if layers == 1:
        model.add(LSTM(units, return_sequences=True, input_shape=(window_size, X_train.shape[2])))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    else:
        model.add(LSTM(units, return_sequences=True, input_shape=(window_size, X_train.shape[2])))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

        for i in range(1, layers - 1):
            model.add(LSTM(units, return_sequences=True))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

        model.add(LSTM(units, return_sequences=True))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Attention
    model.add(Attention())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callback
    )

    # Generate file names
    model_name = f"model_{model_index}_layers{layers}_units{units}_with_attention.h5"
    history_name = f"history_{model_index}.json"
    model_path = os.path.join("models", model_name)
    history_path = os.path.join("histories", history_name)

    # Ensure folders exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("histories", exist_ok=True)

    try:
        # Save model
        model.save(model_path)

        # Save training history
        with open(history_path, "w") as f:
            json.dump(history.history, f)

        # Get best validation loss
        min_val_loss = min(history.history["val_loss"])

        # Write result to CSV
        # result_file = "training_results.csv"
        write_header = not os.path.exists(result_file)

        with open(result_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow([
                    "index","lstm_units", "num_layers", "dropout", "batch_size", "epochs",
                    "optimizer", "model_name", "val_loss", "history_path"
                ])
            writer.writerow([
                model_index,units, layers, dropout_rate, batch_size,
                epochs,  model_name, min_val_loss, history_name
            ])

    except Exception as e:
        print("Error during save or logging:", e)

    return model, history

def evaluate_model(model_path, history_path, X_test, y_test, scaler=None, title=None, attention=False):
    # Load model
    if attention:
        model = load_model(model_path, custom_objects={'Attention': Attention})
    else:
        model  = load_model(model_path)

    # Predict
    y_pred = model.predict(X_test) 
    # Optional: inverse transform
    if y_test.ndim == 1:
        y_test = y_test[:, None]         
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]         

    if scaler:
        min_close, max_close = scaler.data_min_[0], scaler.data_max_[0]
        y_test_plot = y_test * (max_close - min_close) + min_close
        y_pred_plot = y_pred * (max_close - min_close) + min_close
    else:
        y_test_plot = y_test
        y_pred_plot = y_pred

    assert y_test_plot.shape == y_pred_plot.shape
    # Metrics
    mse = mean_squared_error(y_test_plot, y_pred_plot)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_plot, y_pred_plot)
    mape = np.mean(np.abs((y_test_plot - y_pred_plot) / y_test_plot)) * 100
    r2 = r2_score(y_test_plot, y_pred_plot)

    # Load history if available
    train_loss = val_loss = None
    if history_path:
        with open(history_path, "r") as f:
            history = json.load(f)
        train_loss = history.get("loss", None)
        val_loss = history.get("val_loss", None)

    # --- Combined Plot ---
    # fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig, axes = plt.subplots(3, 1, figsize=(12, 12),
                         gridspec_kw={'height_ratios': [3, 2, 1]})

    # ① Prediction vs True
    axes[0].plot(y_test_plot, label="True", color="blue")
    axes[0].plot(y_pred_plot, label="Predicted", color="orange")
    axes[0].set_title(title or "Prediction vs True")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Price" if scaler else "Scaled Value")
    axes[0].legend()
    axes[0].grid(True)

    # ② Loss Curve
    if train_loss and val_loss:
        axes[1].plot(train_loss, label="Train Loss", color="green")
        axes[1].plot(val_loss, label="Validation Loss", color="red")
        axes[1].set_title("Training vs Validation Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].text(0.5, 0.5, "No loss history found", fontsize=12, ha='center')
    axes[2].axis("off")                           
    text_str = (
        f"MSE   : {mse:,.2f}\n"
        f"RMSE  : {rmse:,.2f}\n"
        f"MAE   : {mae:,.2f}\n"
        f"MAPE  : {mape:.2f} %\n"
        f"R²    : {r2:.6f}"
    )
    axes[2].text(0.01, 0.9, "Evaluation Metrics", fontsize=14, weight='bold')
    axes[2].text(0.01, 0.5, text_str, fontsize=12, va='top',
                family="monospace")   
        
    plt.tight_layout()
    plt.show()

    # Print metrics
    # print(f"Model: {model_path}")
    # print(f"MSE:  {mse:.6f}")
    # print(f"RMSE: {rmse:.6f}")
    # print(f"MAE:  {mae:.6f}")
    # print(f"MAPE: {mape:.2f}%")
    # print(f"R²:   {r2:.6f}")

    return {
        "model": model_path,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def evaluate_model_multistep(model_path,
                             history_path,
                             X_test,
                             y_test,
                             future_steps=5,
                             scaler=None,
                             title=None,
                             attention=False):

    if attention:
        model = load_model(model_path, custom_objects={'Attention': Attention})
    else:
        model  = load_model(model_path)
    y_pred  = model.predict(X_test)               

    if scaler is not None:
        min_close, max_close = scaler.data_min_[0], scaler.data_max_[0]
        y_test_plot = y_test * (max_close - min_close) + min_close
        y_pred_plot = y_pred * (max_close - min_close) + min_close
    else:
        y_test_plot = y_test
        y_pred_plot = y_pred

    if y_test_plot.ndim == 1:
        y_test_plot = y_test_plot[:, None]              # (N,1)

   
    mse  = mean_squared_error(y_test_plot, y_pred_plot, multioutput="raw_values")
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test_plot, y_pred_plot, multioutput="raw_values")
    mape = np.mean(np.abs((y_test_plot - y_pred_plot) / y_test_plot), axis=0) * 100
    r2   = [r2_score(y_test_plot[:, k], y_pred_plot[:, k]) for k in range(future_steps)]

    
    fig, ax = plt.subplots(figsize=(14, 6))
    show_n = min(300, len(y_test_plot))
    for k in range(future_steps):
        ax.plot(y_test_plot[:show_n, k], label=f"True t+{k+1}")
        ax.plot(y_pred_plot[:show_n, k], "--", label=f"Pred t+{k+1}")
    ax.set_title(title or f"Multi-step Forecast (first {show_n} samples)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Price" if scaler else "Scaled Value")
    ax.legend(ncol=2); ax.grid(True); plt.show()

    
    if history_path and os.path.exists(history_path):
        with open(history_path, "r") as f:
            hist = json.load(f)
        if "loss" in hist and "val_loss" in hist:
            plt.figure(figsize=(8, 4))
            plt.plot(hist["loss"],     label="Train Loss")
            plt.plot(hist["val_loss"], label="Val Loss")
            plt.title("Training vs Validation Loss")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True); plt.show()

    
    print(f"{'Step':>4} | {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'MAPE%':>8} {'R²':>6}")
    for k in range(future_steps):
        print(f"{k+1:>4} | {mse[k]:10.4f} {rmse[k]:10.4f} {mae[k]:10.4f} {mape[k]:8.2f} {r2[k]:6.4f}")

    return {
        "mse":  mse.tolist(),
        "rmse": rmse.tolist(),
        "mae":  mae.tolist(),
        "mape": mape.tolist(),
        "r2":   r2
    }