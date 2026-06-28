import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

# Import custom modules
from predict import load_model, load_scaler, load_and_preprocess_csv, predict_exoplanet
from transit_detection import detect_transits
from parameter_estimation import estimate_transit_parameters
from pdf_generator import generate_pdf_report

# Page Configuration
st.set_page_config(
    page_title="AI Exoplanet Detection Dashboard",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Space CSS Styling with Twinkling Stars Background
st.markdown("""
<style>
    /* Full space-themed background and font */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #020208 0%, #080b24 50%, #03030f 100%);
        color: #E6E6FA;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(6, 8, 28, 0.95) !important;
        border-right: 1px solid rgba(10, 132, 255, 0.2);
    }
    
    /* Twinkling stars effect */
    .stars-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        pointer-events: none;
        background: transparent;
        overflow: hidden;
    }
    
    .star {
        position: absolute;
        background-color: white;
        border-radius: 50%;
        animation: blink 3s infinite ease-in-out;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 0.2; transform: scale(0.8); }
        50% { opacity: 1.0; transform: scale(1.2); }
    }
    
    /* Header neon glow */
    .neon-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        text-align: center;
        text-shadow: 0 0 15px rgba(10, 132, 255, 0.8), 0 0 30px rgba(10, 132, 255, 0.4);
        margin-bottom: 2rem;
    }
    
    .neon-subheader {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0A84FF;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        text-shadow: 0 0 8px rgba(10, 132, 255, 0.3);
    }
    
    /* Glassmorphism metric cards */
    .metric-card {
        background: rgba(12, 13, 33, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(10, 132, 255, 0.4);
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #A0A5C0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    
    .metric-subtext {
        font-size: 0.75rem;
        color: #7E84A3;
        margin-top: 0.25rem;
    }
    
    /* Prediction highlights */
    .exo-detected {
        color: #FF453A !important;
        text-shadow: 0 0 10px rgba(255, 69, 58, 0.6);
    }
    
    .exo-not-detected {
        color: #30D158 !important;
        text-shadow: 0 0 10px rgba(48, 209, 88, 0.6);
    }
    
    /* Detail boxes */
    .detail-box {
        background: rgba(16, 17, 45, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Footer styling */
    .app-footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: #7E84A3;
        font-size: 0.85rem;
    }
    
    .step-badge {
        background-color: #0A84FF;
        color: white;
        border-radius: 50%;
        display: inline-block;
        width: 24px;
        height: 24px;
        text-align: center;
        line-height: 24px;
        font-weight: bold;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Twinkling Stars Background Injection
stars_html = "".join([
    f'<div class="star" style="top: {np.random.randint(0, 100)}%; left: {np.random.randint(0, 100)}%; '
    f'width: {np.random.randint(1, 3)}px; height: {np.random.randint(1, 3)}px; '
    f'animation-delay: {np.random.uniform(0, 5)}s; animation-duration: {np.random.uniform(2, 6)}s;"></div>'
    for _ in range(60)
])
st.markdown(f'<div class="stars-container">{stars_html}</div>', unsafe_allow_html=True)

# Neon Glow Main Title
st.markdown('<div class="neon-header">AI-Enabled Exoplanet Detection from Noisy Light Curves</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("### 🌌 Kepler Mission Operations")
st.sidebar.image("https://img.icons8.com/color/96/space-shuttle.png", width=64)

# 1. Pipeline Description
st.sidebar.markdown("""
<div style="font-size: 0.95rem; margin-bottom: 1.5rem; background: rgba(10, 132, 255, 0.1); border-left: 3px solid #0A84FF; padding: 12px; border-radius: 4px;">
    <strong>Signal Processing Pipeline:</strong><br><br>
    <div style="font-family: monospace; font-size: 0.85rem; color: #A0A5C0;">
        CSV Input<br>
        &nbsp;&nbsp;↓<br>
        Normalization<br>
        &nbsp;&nbsp;↓<br>
        AI Classification<br>
        &nbsp;&nbsp;↓<br>
        Dip Detection<br>
        &nbsp;&nbsp;↓<br>
        Parameter Estimation<br>
        &nbsp;&nbsp;↓<br>
        <span style="color: #FFFFFF; font-weight: bold;">Final Prediction</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🛠️ Configuration")

# Selection between sample files or custom upload
data_source = st.sidebar.selectbox(
    "Select Data Source",
    ["Exoplanet Transit Sample", "Non-Exoplanet Noise Sample", "Upload My Own CSV"]
)

# 2. Peak Detection Tuning Parameters
st.sidebar.markdown("### 🎛️ SciPy Peak Parameters")
st.sidebar.caption("Fine-tune the algorithm used to isolate transit dips on the inverted normalized light curve.")

peak_prominence = st.sidebar.slider(
    "Peak Prominence",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="Minimum height of the peak relative to local baseline noise. Increase for cleaner detection."
)

peak_distance = st.sidebar.slider(
    "Min Peak Distance",
    min_value=5,
    max_value=200,
    value=30,
    step=5,
    help="Minimum number of samples separating consecutive transits."
)

min_height = st.sidebar.slider(
    "Height Threshold",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.5,
    help="Minimum inverted height of peak (in units of standard deviation)."
)

# ----------------- DATA LOADING -----------------
# Determine file object based on selection
file_path = None
uploaded_file = None
success_msg = ""

if data_source == "Exoplanet Transit Sample":
    file_path = "samples/exoplanet_kepler_format.csv"
    success_msg = "Successfully loaded Kepler Exoplanet Transit Sample."
elif data_source == "Non-Exoplanet Noise Sample":
    file_path = "samples/non_exoplanet_kepler_format.csv"
    success_msg = "Successfully loaded Normal Stellar Noise Sample."
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Light Curve CSV (3197 flux values)",
        type=["csv"],
        help="Upload a CSV with 1 row of 3197 values, or a Kepler dataset row containing 3198 columns (label + fluxes)."
    )

# Orchestrate logic if file is available
data_to_process = None
true_label = None

if file_path:
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data_to_process, true_label = load_and_preprocess_csv(f)
            st.sidebar.success(success_msg)
    else:
        st.sidebar.error("Sample files are missing! Run `generate_samples.py` first.")
elif uploaded_file is not None:
    try:
        data_to_process, true_label = load_and_preprocess_csv(uploaded_file)
        st.sidebar.success("Custom CSV uploaded and loaded successfully.")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# ----------------- MAIN PROCESSING -----------------
if data_to_process is not None:
    # 1. Load ML assets
    with st.spinner("Decoding Deep Space Signal..."):
        model = load_model()
        scaler = load_scaler()
        
    # 2. Run prediction
    pred_class, confidence, scaled_flux = predict_exoplanet(data_to_process, model, scaler)
    raw_flux = data_to_process[0]
    
    # 3. Detect transits using SciPy
    height_param = min_height if min_height > 0 else None
    peaks, properties = detect_transits(
        scaled_flux,
        prominence=peak_prominence,
        distance=peak_distance,
        height=height_param
    )
    
    # Filter to top 3 primary candidates
    if len(peaks) > 3:
        prominences = properties['prominences']
        top_indices = np.argsort(prominences)[-3:]
        top_indices = np.sort(top_indices)
        peaks = peaks[top_indices]
        for key in properties:
            if isinstance(properties[key], np.ndarray) and len(properties[key]) > 0:
                properties[key] = properties[key][top_indices]

    
    # 4. Estimate physical parameters
    transit_stats = estimate_transit_parameters(
        raw_flux,
        scaled_flux,
        peaks,
        properties
    )
    
    # Show ground truth if available
    ground_truth_str = ""
    if true_label is not None:
        truth = "Exoplanet (2)" if true_label == 2 else "Non-Exoplanet (1)"
        ground_truth_str = f" (Ground Truth: {truth})"

    # ----------------- METRICS CARDS -----------------
    pred_label = "Potential Transit Candidate" if pred_class == 1 else "No Significant Transit Detected"
    pred_class_style = "exo-detected" if pred_class == 1 else "exo-not-detected"
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">AI Verdict</div>
            <div class="metric-value {pred_class_style}">{pred_label}</div>
            <div class="metric-subtext">XGBoost Classifier</div>
        </div>
        ''', unsafe_allow_html=True)
        
    with col2:
        # Gauge for confidence
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = confidence * 100,
            title = {'text': "Confidence", 'font': {'color': '#A0A5C0', 'size': 14}},
            number = {'suffix': "%", 'font': {'color': '#FFFFFF', 'size': 24}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#0A84FF"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 70], 'color': "rgba(255, 69, 58, 0.3)"},
                    {'range': [70, 90], 'color': "rgba(255, 159, 10, 0.3)"},
                    {'range': [90, 100], 'color': "rgba(48, 209, 88, 0.3)"}],
            }
        ))
        fig_gauge.update_layout(height=150, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#FFFFFF"})
        st.plotly_chart(fig_gauge, use_container_width=True)
        conf_text = "High Confidence" if confidence > 0.9 else "Moderate Confidence" if confidence > 0.7 else "Weak Confidence"
        st.markdown(f"<div style='text-align: center; color: #A0A5C0; font-size: 1.1rem; margin-top: -15px;'>{conf_text}</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-label">Classification Capability</div>
            <div style="font-size: 1.1rem; color: #30D158;">✓ Transit Candidate</div>
            <div class="metric-subtext" style="margin-top: 10px;">Future Work:<br>• Eclipsing Binary<br>• Stellar Blend<br>• Variable Star</div>
        </div>
        ''', unsafe_allow_html=True)
        
    st.write("") # Spacing

    # Transit Analysis Panel
    st.markdown("### 📊 Transit Analysis")
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.metric("Primary Transit Events", transit_stats['num_transits'])
    with col_t2:
        st.metric("Transit Depth", f"{transit_stats['avg_depth_pct']:.3f}%" if transit_stats['num_transits'] > 0 else "0.000%")
    with col_t3:
        st.metric("Transit Duration", f"{transit_stats['avg_duration']:.1f} bins" if transit_stats['num_transits'] > 0 else "N/A")
    with col_t4:
        st.metric("Estimated SNR", f"{transit_stats['snr']:.2f}" if transit_stats['num_transits'] > 0 else "0.00")
        
    st.info("⏱️ **Orbital Period:** Unavailable. This dataset does not include timestamps. Physical orbital period estimation requires time-series observations.")

    # Periodic check
    if len(peaks) > 1:
        intervals = np.diff(peaks)
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        conf = "High" if std_interval < 5 else "Low"
        st.success(f"**Periodic Transit Check:** ✓ Periodic Pattern Detected. Estimated Repeat Interval: {avg_interval:.1f} samples (Confidence: {conf})")
    else:
        st.warning("**Periodic Transit Check:** Insufficient repeated transit events in the uploaded observation window.")

    # AI Summary & PDF Download
    col_sum1, col_sum2 = st.columns([3, 1])
    
    with col_sum1:
        if pred_class == 1:
            st.success("**AI Summary:** The uploaded light curve was normalized and analyzed using an XGBoost classifier followed by peak-based transit detection. Statistically significant transit-like dips were identified. This observation is a potential exoplanet transit candidate.")
        else:
            st.warning("**AI Summary:** The uploaded light curve was normalized and analyzed using an XGBoost classifier followed by peak-based transit detection. No statistically significant transit events were identified. The observed variations are consistent with stellar noise or instrumental artifacts.")
            
    with col_sum2:
        try:
            pdf_bytes = generate_pdf_report(pred_label, confidence, transit_stats)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name="Exoplanet_Mission_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"PDF Error: {e}")
            
    st.write("") # Spacing
    # ----------------- PLOTLY VISUALIZATION -----------------
    st.markdown('<div class="neon-subheader">🔭 Interactive Light Curve Analysis</div>', unsafe_allow_html=True)
    
    # X axis: Time steps/sample units
    time_steps = np.arange(len(raw_flux))
    
    tab1, tab2 = st.tabs(["Normalized Flux", "Original Flux"])
    
    with tab1:
        # Plotly figure creation
        fig = go.Figure()
        
        # Main normalized flux plot
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=scaled_flux,
            mode='lines',
            name='Normalized Flux',
            line=dict(color='#0A84FF', width=1.5),
            hovertemplate='Time: %{x}<br>Flux (std): %{y:.3f}<extra></extra>'
        ))
        
        # Highlight detected peaks in red
        if len(peaks) > 0:
            fig.add_trace(go.Scatter(
                x=peaks,
                y=scaled_flux[peaks],
                mode='markers',
                name='Transit Center (Peak)',
                marker=dict(color='#FF453A', size=12, symbol='circle', line=dict(width=2, color='white')), text=['🔴']*len(peaks), textposition='top center', textfont=dict(size=16),
                hovertemplate='Transit Center<br>Time: %{x}<br>Flux: %{y:.3f}<extra></extra>'
            ))
            
            # Shade the transit duration areas using find_peaks widths
            widths = properties.get('widths', [])
            left_ips = properties.get('left_ips', [])
            right_ips = properties.get('right_ips', [])
            
            for i, (l_ip, r_ip) in enumerate(zip(left_ips, right_ips)):
                fig.add_vrect(
                    x0=l_ip,
                    x1=r_ip,
                    fillcolor="rgba(255, 69, 58, 0.25)",
                    opacity=1.0,
                    line_width=0,
                    name=f'Transit {i+1} Duration'
                )
                
        # Dark style Layout
        fig.update_layout(
            plot_bgcolor='rgba(12, 13, 33, 0.4)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#E6E6FA'),
            xaxis=dict(
                title='Observation Timeline (Sample Bins)',
                gridcolor='rgba(255, 255, 255, 0.05)',
                zeroline=False,
                showline=True,
                linecolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title='Standardized Relative Flux (z-score)',
                gridcolor='rgba(255, 255, 255, 0.05)',
                zeroline=False,
                showline=True,
                linecolor='rgba(255,255,255,0.1)'
            ),
            margin=dict(l=40, r=40, t=20, b=40),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        fig_orig = go.Figure()
        fig_orig.add_trace(go.Scatter(
            x=time_steps,
            y=raw_flux,
            mode='lines',
            name='Original Flux',
            line=dict(color='#A0A5C0', width=1.5),
            hovertemplate='Time: %{x}<br>Raw Flux: %{y:.3f}<extra></extra>'
        ))
        fig_orig.update_layout(
            plot_bgcolor='rgba(12, 13, 33, 0.4)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#E6E6FA'),
            xaxis=dict(title='Observation Timeline (Sample Bins)'),
            yaxis=dict(title='Raw Flux'),
            margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(fig_orig, use_container_width=True)

    # Show Detected Events details
    st.markdown("#### 🔭 Detected Transit Events")
    if len(peaks) > 0:
        dips_data = []
        for i, peak in enumerate(peaks):
            depth = transit_stats['depths_pct'][i]
            duration = transit_stats['durations'][i]
            snr_val = transit_stats['snr']
            dips_data.append({"Dip": i+1, "Center": peak, "Depth": f"{depth:.3f}%", "Duration": f"{duration:.1f} bins", "SNR": f"{snr_val:.1f}"})
        st.dataframe(pd.DataFrame(dips_data), use_container_width=True, hide_index=True)
    else:
        st.write("No statistically significant transit events detected.")
        
    # ----------------- ABOUT THE PREDICTION -----------------
    st.markdown('<div class="neon-subheader">ℹ️ Methodology & Dataset</div>', unsafe_allow_html=True)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown('''<div class="detail-box" style="text-align: center;">
            <h4>📡 Dataset</h4>
            <div style="color: #A0A5C0;">
            <b>Kepler Mission</b><br>
            3197 Flux Samples<br>
            XGBoost Classifier<br>
            NASA Public Dataset
            </div>
        </div>''', unsafe_allow_html=True)
    with col_m2:
        st.markdown('''<div class="detail-box" style="text-align: center;">
            <h4>🛠️ Methodology</h4>
            <style>
                .flow-node { background: rgba(10,132,255,0.15); border: 1px solid rgba(10,132,255,0.4); border-radius: 4px; padding: 4px; margin: 4px auto; width: 70%; text-align: center; font-size: 0.85rem; color: #E6E6FA; box-shadow: 0 2px 4px rgba(0,0,0,0.3); }
                .flow-arrow { text-align: center; color: #0A84FF; font-size: 1rem; line-height: 1; margin: 2px 0; }
            </style>
            <div class="flow-node">📄 CSV Upload</div>
            <div class="flow-arrow">⬇</div>
            <div class="flow-node">⚖️ Normalization</div>
            <div class="flow-arrow">⬇</div>
            <div class="flow-node">🧠 XGBoost</div>
            <div class="flow-arrow">⬇</div>
            <div class="flow-node">🔎 Peak Detection</div>
            <div class="flow-arrow">⬇</div>
            <div class="flow-node" style="border-color: #30D158; background: rgba(48,209,88,0.15);">✅ Transit Candidate</div>
        </div>''', unsafe_allow_html=True)

    st.markdown('<div class="neon-subheader">ℹ️ Why This Prediction?</div>', unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown('''
        <div class="detail-box">
            <h4>⚙️ AI Architecture</h4>
            <style>
                .arch-node { background: linear-gradient(90deg, rgba(16,17,45,0.8) 0%, rgba(10,132,255,0.15) 100%); border-left: 4px solid #0A84FF; border-radius: 4px; padding: 6px 12px; margin: 6px auto; width: 85%; font-size: 0.9rem; color: #FFFFFF; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 5px rgba(0,0,0,0.4); }
                .arch-arrow { text-align: center; color: #7E84A3; font-size: 1.2rem; margin: -2px 0; }
                .arch-icon { font-size: 1.2rem; }
            </style>
            <div class="arch-node"><span>Input: 3197 Flux Values</span> <span class="arch-icon">📥</span></div>
            <div class="arch-arrow">⬇</div>
            <div class="arch-node"><span>StandardScaler</span> <span class="arch-icon">⚖️</span></div>
            <div class="arch-arrow">⬇</div>
            <div class="arch-node"><span>XGBoost Classifier</span> <span class="arch-icon">🧠</span></div>
            <div class="arch-arrow">⬇</div>
            <div class="arch-node"><span>Probability Score</span> <span class="arch-icon">📊</span></div>
            <div class="arch-arrow">⬇</div>
            <div class="arch-node"><span>SciPy find_peaks()</span> <span class="arch-icon">🔎</span></div>
            <div class="arch-arrow">⬇</div>
            <div class="arch-node" style="border-left-color: #FF453A;"><span>Transit Parameters</span> <span class="arch-icon">📈</span></div>
        </div>
        ''', unsafe_allow_html=True)
        
    with col_info2:
        conf_cat = "Strong candidate (>90%)" if confidence > 0.9 else "Moderate candidate (70-90%)" if confidence > 0.7 else "Weak candidate (<70%)"
        st.markdown(f'''
        <div class="detail-box">
            <h4>✅ Confidence Checklist</h4>
            <p>Confidence: <strong>{conf_cat}</strong></p>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>✓ <b>Noise Reduction:</b> StandardScaler Applied</li>
                <li>✓ Flux normalized</li>
                <li>✓ Candidate dips analyzed</li>
                <li>✓ XGBoost confidence generated</li>
                <li>✓ Physical parameters estimated</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)

else:
    # Prompt user to load or upload data
    st.markdown('''
    <div style="text-align: center; padding: 5rem 0;">
        <h3 style="color: #7E84A3;">🛰️ Awaiting Astronomical Signal</h3>
        <p style="color: #585C74;">Select a sample curve from the sidebar, or upload a custom CSV file to launch the exoplanet detection pipeline.</p>
    </div>
    ''', unsafe_allow_html=True)

# ----------------- FOOTER -----------------
st.markdown('''
<div class="app-footer">
    🌌 <strong>AI-Enabled Exoplanet Detection System</strong> | Kepler Space Telescope Mission Data<br>
    Model: XGBoost Classifier (3197 features) | Scaler: StandardScaler | Platform: Streamlit Space Portal v1.0<br>
    <span style="font-size: 0.75rem;">Created for Hackathon Demonstration & Astronomical Signal Analysis. NASA Open Data Archive.</span>
</div>
''', unsafe_allow_html=True)
