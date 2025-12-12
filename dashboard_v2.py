import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# --- PAGE CONFIGURATION (The "Vibe") ---
st.set_page_config(
    page_title="GridGuard AI 3000",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD RESOURCES ---
model_path = 'grid_fault_ai_model.pkl'
try:
    clf = joblib.load(model_path)
except:
    st.error("🚨 BRAIN MISSING! Run training script first.")
    st.stop()

fault_map = {
    0: 'System Healthy 🟢', 
    1: 'Single L-G (Phase A) ⚡', 2: 'Single L-G (Phase B) ⚡', 3: 'Single L-G (Phase C) ⚡',
    4: 'Line-to-Line (A-B) 🔥', 5: 'Line-to-Line (B-C) 🔥', 6: 'Line-to-Line (C-A) 🔥',
    7: 'Double L-G (AB-G) 💥', 8: 'Double L-G (BC-G) 💥', 9: 'Double L-G (CA-G) 💥',
    10: 'Three-Phase (ABC) ☠️', 11: 'Three-Phase-Ground (ABC-G) ☠️'
}

# --- CUSTOM CSS (Making it look cool) ---
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #dcdcdc;
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ GridGuard AI Command Center")
st.markdown("Welcome, **Operator**. Don't blow up the grid today, okay?")

# --- SIDEBAR: THE CONTROL PANEL ---
st.sidebar.header("🎛️ Control Panel")

# 1. Chaos Button (Randomizer)
if st.sidebar.button("🎲 CHAOS MODE (Random Fault)"):
    # Generate random crazy values
    def_va, def_vb, def_vc = np.random.randint(0, 7000, 3)
    def_ia, def_ib, def_ic = np.random.randint(10, 5000, 3)
    st.toast("⚠️ Injecting Chaos into the system...", icon="😈")
else:
    # Default healthy values
    def_va, def_vb, def_vc = 6350.0, 6350.0, 6350.0
    def_ia, def_ib, def_ic = 15.0, 15.0, 15.0

# 2. Sliders (More interactive than typing)
st.sidebar.subheader("Voltage Regulator (V)")
va = st.sidebar.slider("Phase A Voltage", 0.0, 7000.0, float(def_va))
vb = st.sidebar.slider("Phase B Voltage", 0.0, 7000.0, float(def_vb))
vc = st.sidebar.slider("Phase C Voltage", 0.0, 7000.0, float(def_vc))

st.sidebar.subheader("Current Injection (A)")
ia = st.sidebar.slider("Phase A Current", 0.0, 5000.0, float(def_ia))
ib = st.sidebar.slider("Phase B Current", 0.0, 5000.0, float(def_ib))
ic = st.sidebar.slider("Phase C Current", 0.0, 5000.0, float(def_ic))

# --- REAL-TIME VISUALIZATION (The "Pro" Look) ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📡 Live Waveform Monitor")
    # Generate fake sine waves based on inputs for visualization
    t = np.linspace(0, 0.04, 100) # 2 cycles at 50Hz
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 3))
    # Scale peaks by sqrt(2) approx
    ax.plot(t, va * 1.414 * np.sin(2*np.pi*50*t), color='red', label='Phase A')
    ax.plot(t, vb * 1.414 * np.sin(2*np.pi*50*t - 2*np.pi/3), color='green', label='Phase B')
    ax.plot(t, vc * 1.414 * np.sin(2*np.pi*50*t + 2*np.pi/3), color='blue', label='Phase C')
    ax.set_title("Voltage Oscilloscope")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    st.pyplot(fig)

with col2:
    st.subheader("🧭 Phasor Radar")
    # Simple Polar Plot
    fig_p, ax_p = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # Angles for A, B, C
    angles = [0, -2*np.pi/3, 2*np.pi/3]
    magnitudes = [va, vb, vc]
    colors = ['r', 'g', 'b']
    
    for ang, mag, c in zip(angles, magnitudes, colors):
        ax_p.annotate("", xy=(ang, mag), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="->", color=c, lw=3))
    
    ax_p.set_yticklabels([]) # Hide radial labels
    st.pyplot(fig_p)

# --- AI DIAGNOSIS ENGINE ---
st.divider()

# Prepare Data
i_res = np.sqrt(((ia + ib + ic)**2)/3) # Approx Zero Sequence
inputs = [i_res, va, vb, vc, ia, ib, ic, va*1.414, vb*1.414, vc*1.414, ia*1.414, ib*1.414, ic*1.414]
cols = ['I_Residual', 'Va_RMS', 'Vb_RMS', 'Vc_RMS', 'Ia_RMS', 'Ib_RMS', 'Ic_RMS',
        'Va_Peak', 'Vb_Peak', 'Vc_Peak', 'Ia_Peak', 'Ib_Peak', 'Ic_Peak']
input_df = pd.DataFrame([inputs], columns=cols)

# Predict
prediction = clf.predict(input_df)[0]
probs = clf.predict_proba(input_df)[0]
confidence = np.max(probs) * 100
diagnosis = fault_map.get(prediction, "Unknown")

# --- THE FUN PART: RESULTS & MEMES ---
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("AI Diagnosis Report")
    if prediction == 0:
        st.success(f"System Status: {diagnosis}")
        st.balloons() # Fun effect for healthy system
    else:
        st.error(f"System Status: {diagnosis}")
    
    st.metric(label="AI Confidence Level", value=f"{confidence:.2f}%")
    st.info(f"Residual Current Detected: {i_res:.2f} A")

with c2:
    st.subheader("Operator Reaction Cam")
    
    # Funny logic for images
    if prediction == 0:
        st.image("https://media.giphy.com/media/11sBLVxNs7v6WA/giphy.gif", caption="Grid is chilling.")
    elif i_res > 1000:
        st.image("https://media.giphy.com/media/OE6FE4G8V8K4/giphy.gif", caption="GROUND FAULT! Earth is angry!")
    elif prediction in [10, 11]:
        st.image("https://media.giphy.com/media/NTur7XlVDUdqM/giphy.gif", caption="TOTAL BLACKOUT IMMINENT!")
    else:
        st.image("https://media.giphy.com/media/HUkOv6BNWc1HO/giphy.gif", caption="Something is broken...")

# --- FOOTER ---
st.markdown("---")
st.caption("GridGuard AI v1.0 | Built with Simulink & Python | 'Because Manual Breakers are for Amateurs'")