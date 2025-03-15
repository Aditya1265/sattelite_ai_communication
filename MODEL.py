import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import json

# Load Preprocessed Data
X_train = pd.read_csv("X_train_preprocessed.csv").values
X_test = pd.read_csv("X_test_preprocessed.csv").values
y_train = pd.read_csv("y_train.csv").values.ravel()  # Ensure correct shape
y_test = pd.read_csv("y_test.csv").values.ravel()

# Normalize Data (if not already normalized)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use in Streamlit
joblib.dump(scaler, "scaler.pkl")

# Define Deep Learning Model
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Sigmoid keeps output in [0,1]
])

# Compile Model
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=0.0001), loss='mse', metrics=['mae'])

# Train Model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Save Model
model.save("satellite_nn_model.h5")

# Save Training History
with open("training_history.json", "w") as f:
    json.dump(history.history, f)

print("✅ Model and Preprocessing Pipeline Saved Successfully!")


# Load training history
history_dict = history.history  # Assuming you have a history object from model.fit()

# Plot loss
plt.plot(history_dict['loss'], label='Training Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()
