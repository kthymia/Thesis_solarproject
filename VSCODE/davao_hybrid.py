import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
DATA_PATH = 'NASA-POWER_CLEANED.csv'  # Path to your cleaned CSV
SEQUENCE_LENGTH = 30  # Number of days in each input sequence
FEATURES = [
    'T2M',              # Air temperature at 2m (°C)
    'RH2M',             # Relative humidity at 2m (%)
    'WS2M',             # Wind speed at 2m (m/s)
    'ALLSKY_SFC_SW_DWN',# Surface shortwave downwelling (target)
    'PRECTOTCORR',      # Corrected precipitation (mm)
]
TARGET = 'ALLSKY_SFC_SW_DWN'  # Solar irradiance (target)
TEST_SIZE = 0.2  # 80/20 train/test split
RANDOM_STATE = 42

# --- 1. LOAD DATA ---
df = pd.read_csv(DATA_PATH)
assert all(f in df.columns for f in FEATURES), f"Missing features in CSV: {set(FEATURES) - set(df.columns)}"
data = df[FEATURES].copy()

# --- 2. FEATURE SCALING ---
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# --- 3. CREATE SEQUENCES ---
def create_sequences(data, seq_length, target_col_idx):
    X_seq, X_ann, y = [], [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length, :]
        X_seq.append(seq)
        X_ann.append(seq[-1, :])  # last day's features
        y.append(data[i+seq_length, target_col_idx])
    return np.array(X_seq), np.array(X_ann), np.array(y)

X_seq, X_ann, y = create_sequences(data_scaled, SEQUENCE_LENGTH, FEATURES.index(TARGET))

# --- 4. TRAIN/TEST SPLIT ---
split_idx = int((1 - TEST_SIZE) * len(X_seq))
X_seq_train, X_seq_test = X_seq[:split_idx], X_seq[split_idx:]
X_ann_train, X_ann_test = X_ann[:split_idx], X_ann[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- 5. MODEL DEFINITION (HYBRID) ---
# LSTM branch
lstm_input = Input(shape=(SEQUENCE_LENGTH, len(FEATURES)), name='lstm_input')
lstm_x = LSTM(50, return_sequences=True)(lstm_input)
lstm_x = LSTM(30)(lstm_x)

# ANN branch
ann_input = Input(shape=(len(FEATURES),), name='ann_input')
ann_x = Dense(32, activation='relu')(ann_input)
ann_x = Dense(16, activation='relu')(ann_x)

# Concatenate
concat = Concatenate()([lstm_x, ann_x])
output = Dense(1)(concat)

model = Model(inputs=[lstm_input, ann_input], outputs=output)
model.compile(optimizer='adam', loss='mse')

# --- 6. TRAINING ---
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    [X_seq_train, X_ann_train], y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# --- 7. PREDICTION & INVERSE SCALING ---
y_pred = model.predict([X_seq_test, X_ann_test])

# Inverse scale the predictions and true values (only the target column)
pad_shape = (len(y_pred), len(FEATURES))
y_pred_pad = np.zeros(pad_shape)
y_test_pad = np.zeros(pad_shape)
y_pred_pad[:, FEATURES.index(TARGET)] = y_pred.flatten()
y_test_pad[:, FEATURES.index(TARGET)] = y_test.flatten()
inv_y_pred = scaler.inverse_transform(y_pred_pad)[:, FEATURES.index(TARGET)]
inv_y_test = scaler.inverse_transform(y_test_pad)[:, FEATURES.index(TARGET)]

# --- 8. EVALUATION ---
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

print("\n--- Evaluation on Test Set ---")
print(f"R^2:   {r2_score(inv_y_test, inv_y_pred):.4f}")
print(f"RMSE:  {np.sqrt(mean_squared_error(inv_y_test, inv_y_pred)):.4f}")
print(f"MAE:   {mean_absolute_error(inv_y_test, inv_y_pred):.4f}")
print(f"MAPE:  {mape(inv_y_test, inv_y_pred):.2f}%")

# --- 9. PLOTS ---
fig, axs = plt.subplots(2, 2, figsize=(16, 10))

# Training & Validation Loss
axs[0, 0].plot(history.history['loss'], label='Train Loss')
axs[0, 0].plot(history.history['val_loss'], label='Val Loss')
axs[0, 0].set_title('Training & Validation Loss')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('MSE Loss')
axs[0, 0].legend()

# Scatter plot: Predicted vs Actual
sns.scatterplot(x=inv_y_test, y=inv_y_pred, alpha=0.6, ax=axs[0, 1])
axs[0, 1].plot([inv_y_test.min(), inv_y_test.max()], [inv_y_test.min(), inv_y_test.max()], 'r--')
axs[0, 1].set_xlabel('Actual Irradiance')
axs[0, 1].set_ylabel('Predicted Irradiance')
axs[0, 1].set_title('Predicted vs Actual (Test Set)')

# Line plot: Actual vs Predicted (first 200 samples)
axs[1, 0].plot(inv_y_test[:200], label='Actual')
axs[1, 0].plot(inv_y_pred[:200], label='Predicted')
axs[1, 0].set_title('Actual vs Predicted Irradiance (First 200 Test Samples)')
axs[1, 0].set_xlabel('Sample')
axs[1, 0].set_ylabel('Irradiance')
axs[1, 0].legend()

# Residuals histogram
residuals = inv_y_test - inv_y_pred
sns.histplot(residuals, bins=30, kde=True, ax=axs[1, 1])
axs[1, 1].set_title('Residuals Distribution (Test Set)')
axs[1, 1].set_xlabel('Residual (Actual - Predicted)')

plt.tight_layout()
plt.show()

# --- 10. SAVE MODEL ---
model.save('davao_hybrid_model.h5')