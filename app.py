import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

# 🚀 Load Custom CSS for Better UI
st.markdown("""
    <style>
    body {background-color: #f5f5f5;}
    .stApp {background: linear-gradient(to bottom, #ffffff, #e6f7ff);}
    .stTitle {text-align: center; color: #0056b3; font-size: 40px; font-weight: bold;}
    .stSidebar {background-color: #f8f9fa;}
    .stButton>button {background-color: #007bff; color: white; border-radius: 8px; font-size: 16px;}
    </style>
""", unsafe_allow_html=True)

# 🚀 Load the ML Model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("satellite_nn_model.h5", compile=False)
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001), loss="mse", metrics=["mae"])
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.error("🚨 Model failed to load. Please check 'satellite_nn_model.h5'.")

# 🚀 Load MinMaxScaler Parameters
y_min = np.load("y_scaler_min.npy")
y_max = np.load("y_scaler_max.npy")
y_scaler = MinMaxScaler()
y_scaler.min_, y_scaler.scale_ = y_min, 1 / (y_max - y_min)

# ✅ Load Preprocessor (`preprocessor.pkl`)
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# 🚀 Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Prediction", "📈 Data Visualization"])

# 🚀 Home Page
if page == "🏠 Home":
    st.markdown('<h1 class="stTitle">🚀 Satellite AI Model</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="stMarkdown">Designed by <b>Aditya Pandey</b> for AI-powered Satellite Communication</h3>', unsafe_allow_html=True)

    # ✅ Restored Logos & Images
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("assets/Birla_Institute_of_Technology_Mesra.png", width=200)
        st.markdown("### BIT Mesra")
        st.markdown("#### AI-Powered Satellite Communication System")
    
    with col2:
        st.image("assets/917923.png", use_container_width=True)

    st.markdown("""
    ### 📡 Welcome to **Satellite AI Model**
    - **Designed by Aditya Pandey** for satellite communication performance prediction.
    - This model helps in **real-time signal strength prediction** based on user inputs.
    """)

# 📊 Prediction Page
elif page == "📊 Prediction":
    st.title("📊 Satellite Communication Prediction")
    st.markdown("### **Enter Satellite Parameters Below:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        frequency = st.number_input("Frequency (GHz)", min_value=1.5, max_value=30.0, step=0.1)
        bandwidth = st.number_input("Bandwidth (MHz)", min_value=10.0, max_value=500.0, step=1.0)
    
    with col2:
        noise_level = st.number_input("Noise Level (dB)", min_value=-100.0, max_value=-50.0, step=1.0)
        weather_condition = st.selectbox("Weather Condition", ["Clear", "Cloudy", "Rainy", "Stormy"])
    
    with col3:
        location_type = st.selectbox("Location Type", ["Oceanic", "Equatorial", "Polar", "Mountainous", "Mid-Latitude"])
        latency = st.number_input("Latency (ms)", min_value=100.0, max_value=600.0, step=1.0)
        packet_loss = st.number_input("Packet Loss (%)", min_value=0.0, max_value=10.0, step=0.1)

    # 🔹 Convert Inputs to DataFrame
    input_df = pd.DataFrame([[frequency, bandwidth, noise_level, latency, packet_loss, 
                              weather_condition, modulation_scheme, location_type]],
                            columns=["Frequency_GHz", "Bandwidth_MHz", "Noise_Level_dB", "Latency_ms", 
                                     "Packet_Loss_%", "Weather_Condition", "Modulation_Scheme", "Location_Type"])

    # 🔹 Apply Preprocessing
    try:
        input_processed = preprocessor.transform(input_df)
        input_processed = np.array(input_processed).reshape(1, -1)  # Ensure correct shape
        st.write(f"✅ Processed Input Shape: {input_processed.shape}")  # Debugging line
    except Exception as e:
        st.error(f"🚨 Error processing input: {e}")
        st.stop()

    # Prediction Button
    if st.button("🚀 Predict Now") and model is not None:
        try:
            prediction = model.predict(input_processed)
            predicted_signal_strength = y_scaler.inverse_transform(prediction)[0][0]
            st.session_state.prediction = predicted_signal_strength
            st.success(f"📡 **Predicted Signal Strength:** {predicted_signal_strength:.2f} dBm")
        except Exception as e:
            st.error(f"🚨 Prediction Error: {e}")

# 📈 Graphs Page
elif page == "📈 Data Visualization":
    st.title("📈 Data Visualization & Predictions")

    if "prediction" in st.session_state:
        st.markdown("## 📊 User Input Parameters")
        user_data = {
            "Frequency (GHz)": st.session_state.frequency,
            "Bandwidth (MHz)": st.session_state.bandwidth,
            "Noise Level (dB)": st.session_state.noise_level,
            "Weather": st.session_state.weather_condition,
            "Location": st.session_state.location_type,
            "Latency (ms)": st.session_state.latency,
            "Packet Loss (%)": st.session_state.packet_loss
        }
        df = pd.DataFrame(user_data.items(), columns=["Parameter", "Value"])
        
        fig, ax = plt.subplots()
        sns.barplot(x="Parameter", y="Value", data=df, ax=ax, palette="coolwarm")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # 🌀 3D Animated Surface Plot
        st.markdown("### 🌍 3D Animated Signal Strength Visualization")
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')
        X = np.linspace(1.5, 30, 100)
        Y = np.linspace(10, 500, 100)
        X, Y = np.meshgrid(X, Y)
        Z = np.sin(X) * np.cos(Y) * 5 + st.session_state.prediction

        ax.plot_surface(X, Y, Z, cmap="coolwarm")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Bandwidth (MHz)")
        ax.set_zlabel("Signal Strength (dBm)")
        ax.set_title("Predicted Signal Strength Over Time")

        st.pyplot(fig)
    else:
        st.warning("⚠️ No prediction made yet. Please go to the 'Prediction' page first.")
