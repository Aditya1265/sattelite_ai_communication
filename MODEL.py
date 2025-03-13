import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

# Load Preprocessed Data
X_train = pd.read_csv("X_train_preprocessed.csv").values
X_test = pd.read_csv("X_test_preprocessed.csv").values
y_train = pd.read_csv("y_train.csv").values
y_test = pd.read_csv("y_test.csv").values

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
    layers.Dense(1, activation='sigmoid')  # Sigmoid to keep output in [0,1]
])

# Compile Model
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=0.0001), loss='mse', metrics=['mae'])

# Train Model
history = model.fit(X_train, y_train, epochs=150, batch_size=64, validation_data=(X_test, y_test))

# Save Model
model.save("satellite_nn_model.h5")
