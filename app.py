import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib  # ✅ Use joblib for consistency
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

# 🚀 Load Custom CSS for Better UI
st.markdown("""
    <style>
    body {background-color: #f5f5f5;}
    .stApp {background: linear-gradient(to bottom, #ffffff, #e6f7ff);}
    .stTitle {text-align: center; color: #0056b3; font-size: 40px; font-weight: bold;}
    .stSidebar {background-color: #f8f9fa;}
    .stButton>button {background-color: #007bff; color: white; border-radius: 10px;}
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

# ✅ Load MinMaxScaler for Target Variable
try:
    y_scaler = joblib.load("y_scaler.pkl")  # ✅ Load correctly saved scaler
except Exception as e:
    st.error(f"🚨 Error loading 'y_scaler.pkl': {e}")
    y_scaler = None

# ✅ Load and Verify Preprocessor (`preprocessor.pkl`)
try:
    preprocessor = joblib.load("preprocessor.pkl")  # ✅ Use joblib instead of pickle
    if not isinstance(preprocessor, ColumnTransformer):
        st.error("🚨 'preprocessor.pkl' is invalid. Please re-save it.")
        preprocessor = None
except Exception as e:
    st.error(f"🚨 Error loading preprocessor: {e}")
    preprocessor = None

# 🚀 Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Prediction", "📈 Data Visualization"])

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
    ### 📡 Welcome to the **Satellite AI Model**
    - This AI system predicts **real-time satellite signal strength** based on communication parameters.
    - Designed to improve **satellite communication reliability** in various weather & location conditions.
    """)

# 🚀 Prediction Page
if page == "📊 Prediction":
    st.markdown('<h1 class="stTitle">📊 Predict Satellite Signal Strength</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        frequency = st.number_input("Frequency (GHz)", min_value=1.5, max_value=30.0, step=0.1)
        bandwidth = st.number_input("Bandwidth (MHz)", min_value=10.0, max_value=500.0, step=1.0)
        noise_level = st.number_input("Noise Level (dB)", min_value=-100.0, max_value=-50.0, step=1.0)
    
    with col2:
        latency = st.number_input("Latency (ms)", min_value=100.0, max_value=600.0, step=1.0)
        packet_loss = st.number_input("Packet Loss (%)", min_value=0.0, max_value=10.0, step=0.1)
    
    weather_condition = st.selectbox("Weather Condition", ["Clear", "Cloudy", "Rainy", "Stormy"])
    modulation_scheme = st.selectbox("Modulation Scheme", ["QPSK", "8PSK", "16QAM"])
    location_type = st.selectbox("Location Type", ["Oceanic", "Equatorial", "Polar", "Mountainous", "Mid-Latitude"])

    input_df = pd.DataFrame([[frequency, bandwidth, noise_level, latency, packet_loss, 
                              weather_condition, modulation_scheme, location_type]],
                            columns=["Frequency_GHz", "Bandwidth_MHz", "Noise_Level_dB", "Latency_ms", "Packet_Loss_%", 
                                     "Weather_Condition", "Modulation_Scheme", "Location_Type"])

    if preprocessor is not None:
        try:
            input_processed = preprocessor.transform(input_df)
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")
            input_processed = None
    else:
        input_processed = None

    if st.button("🚀 Predict Now") and model is not None and input_processed is not None:
        st.session_state.prediction = y_scaler.inverse_transform(model.predict(input_processed))[0][0]
        st.success(f"📡 **Predicted Signal Strength:** {st.session_state.prediction:.2f} dBm")
    else:
        st.warning("⚠️ Please ensure all inputs are correctly filled.")

# 🚀 Data Visualization Page
elif page == "📈 Data Visualization":
    st.markdown('<h1 class="stTitle">📈 Data Insights & Visualizations</h1>', unsafe_allow_html=True)

    if "prediction" in st.session_state and st.session_state.prediction is not None:
        st.markdown("### 📊 Feature Contribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=["Frequency", "Bandwidth", "Noise", "Latency", "Packet Loss"],
                    y=[st.session_state.get("frequency", 0),
                       st.session_state.get("bandwidth", 0),
                       st.session_state.get("noise_level", 0),
                       st.session_state.get("latency", 0),
                       st.session_state.get("packet_loss", 0)],
                    ax=ax, palette="coolwarm")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.markdown("### 🌍 3D Animated Signal Strength Visualization")
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        X = np.linspace(0, 10, 100)
        Y = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(X, Y)
        Z = np.sin(X) * np.cos(Y) * 5 + st.session_state.prediction

        ax.plot_surface(X, Y, Z, cmap="coolwarm")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency (GHz)")
        ax.set_zlabel("Signal Strength (dBm)")
        ax.set_title("3D Signal Strength Animation")
        st.pyplot(fig)
    else:
        st.warning("⚠️ No prediction made yet. Please go to the 'Prediction' page first.")
