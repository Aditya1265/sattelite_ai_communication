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
    st.markdown('<h3 class="stMarkdown">Designed by <b>Aditya Pandey Student of 2nd year ECE dept at BIT Mesra</b> for AI-powered Satellite Communication</h3>', unsafe_allow_html=True)

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

    if st.button("🚀 Predict Now") and model is not None:
        input_processed = preprocessor.transform(input_df)
        st.session_state.prediction = y_scaler.inverse_transform(model.predict(input_processed))[0][0]
        st.session_state.features = [frequency, bandwidth, noise_level, latency, packet_loss]  # ✅ Store features
        st.success(f"📡 **Predicted Signal Strength:** {st.session_state.prediction:.2f} dBm")

# 🚀 Data Visualization Page
if "📈 Data Visualization" in st.session_state:
    st.markdown('<h1 class="stTitle">📈 Data Insights & Visualizations</h1>', unsafe_allow_html=True)
    if "features" in st.session_state:
        st.markdown("### 📊 Feature Contribution")
        fig, ax = plt.subplots()
        ax.bar(["Frequency", "Bandwidth", "Noise", "Latency", "Packet Loss"],
               st.session_state.features, color="skyblue")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.heatmap(pd.DataFrame([st.session_state.features], columns=["Frequency", "Bandwidth", "Noise", "Latency", "Packet Loss"]).corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        z = np.sin(x * np.pi) * np.cos(y * np.pi)
        ax.plot_surface(x, y, z, cmap='viridis')
        st.pyplot(fig)

        # Animated Sinusoidal Wave
        fig, ax = plt.subplots()
        x = np.linspace(0, 2 * np.pi, 100)
        line, = ax.plot(x, np.sin(x), 'r')
        ax.set_title("Real-time Signal Power Animation")
        ax.set_ylim(-1.5, 1.5)
        
        def update(frame):
            line.set_ydata(np.sin(x + frame / 10.0))
            return line,
        
        ani = animation.FuncAnimation(fig, update, frames=100, interval=50)
        st.pyplot(fig)
    else:
        st.warning("⚠️ No prediction data available. Please make a prediction first.")
