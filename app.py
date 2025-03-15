# 🚀 Data Visualization Page
if "📈 Data Visualization" in st.session_state:
    st.markdown('<h1 class="stTitle">📈 Data Insights & Visualizations</h1>', unsafe_allow_html=True)
    if "features" in st.session_state:
        st.markdown("### 📊 Feature Contribution")
        
        # Bar Chart
        fig, ax = plt.subplots()
        ax.bar(["Frequency", "Bandwidth", "Noise", "Latency", "Packet Loss"],
               st.session_state.features, color="skyblue")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Heatmap
        fig, ax = plt.subplots()
        feature_df = pd.DataFrame([st.session_state.features], 
                                  columns=["Frequency", "Bandwidth", "Noise", "Latency", "Packet Loss"])
        sns.heatmap(feature_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        # 3D Surface Plot
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
