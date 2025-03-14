import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

# Load Dataset
df = pd.read_csv("satellite_communication_optimized_large.csv")

# Drop missing values
df.dropna(inplace=True)

# Define Features and Target
categorical_features = ["Weather_Condition", "Modulation_Scheme", "Location_Type"]
numerical_features = ["Frequency_GHz", "Bandwidth_MHz", "Noise_Level_dB", "Latency_ms", "Packet_Loss_%"]
target = "Signal_Strength_dBm"

# Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Split Data
X = df.drop(columns=[target])
y = df[target].values.reshape(-1, 1)  # Convert target to column vector
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Normalize Target Variable using MinMaxScaler (0-1)
y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

# Save Preprocessed Data
pd.DataFrame(X_train).to_csv("X_train_preprocessed.csv", index=False)
pd.DataFrame(X_test).to_csv("X_test_preprocessed.csv", index=False)
pd.DataFrame(y_train).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

# ✅ Save Preprocessing Pipeline to `.pkl`
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f, protocol=4) 
# ✅ Save Scaler Parameters
np.save("y_scaler_min.npy", y_scaler.data_min_)
np.save("y_scaler_max.npy", y_scaler.data_max_)

print("✅ Preprocessing pipeline saved as 'preprocessor.pkl'")
