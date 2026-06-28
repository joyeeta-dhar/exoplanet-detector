import re
import pandas as pd
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Edit 1: Filter peaks
filter_peaks_code = """    # 3. Detect transits using SciPy
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
"""
content = content.replace("""    # 3. Detect transits using SciPy
    height_param = min_height if min_height > 0 else None
    peaks, properties = detect_transits(
        scaled_flux,
        prominence=peak_prominence,
        distance=peak_distance,
        height=height_param
    )""", filter_peaks_code)

# Edit 2: Rename Prediction -> AI Verdict
content = content.replace('<div class="metric-label">Prediction</div>', '<div class="metric-label">AI Verdict</div>')

# Edit 3: Confidence meter text
confidence_meter_old = "st.plotly_chart(fig_gauge, use_container_width=True)"
confidence_meter_new = """st.plotly_chart(fig_gauge, use_container_width=True)
        conf_text = "High Confidence" if confidence > 0.9 else "Moderate Confidence" if confidence > 0.7 else "Weak Confidence"
        st.markdown(f"<div style='text-align: center; color: #A0A5C0; font-size: 1.1rem; margin-top: -15px;'>{conf_text}</div>", unsafe_allow_html=True)"""
content = content.replace(confidence_meter_old, confidence_meter_new)

# Edit 4: Transit Analysis
content = content.replace('st.metric("Detected Dips", transit_stats[\'num_transits\'])', 'st.metric("Primary Transit Events", transit_stats[\'num_transits\'])')
content = content.replace('st.metric("Depth", f"{transit_stats[\'avg_depth_pct\']:.3f}%"', 'st.metric("Transit Depth", f"{transit_stats[\'avg_depth_pct\']:.3f}%"')
content = content.replace('st.metric("Duration", f"{transit_stats[\'avg_duration\']:.1f} bins"', 'st.metric("Transit Duration", f"{transit_stats[\'avg_duration\']:.1f} bins"')

# Edit 5: AI Summary
content = content.replace("This observation is a strong candidate for an exoplanet transit.", "This observation is a potential exoplanet transit candidate.")

# Edit 6: Table for detected dips
old_dips_display = """    # Show Detected Events details
    if len(peaks) > 0:
        st.markdown("#### 🔭 Detected Transit Events")
        for i, peak in enumerate(peaks):
            depth = transit_stats['depths_pct'][i]
            duration = transit_stats['durations'][i]
            snr_val = transit_stats['snr']
            st.markdown(f"**Event {i+1}** - Center: `{peak}` | Depth: `{depth:.3f}%` | Duration: `{duration:.1f} samples` | SNR: `{snr_val:.1f}`")
    else:
        st.markdown("#### 🔭 Detected Transit Events")
        st.write("No statistically significant transit events detected.")"""
new_dips_display = """    # Show Detected Events details
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
        st.write("No statistically significant transit events detected.")"""
content = content.replace(old_dips_display, new_dips_display)

# Edit 7: Graph marker
content = content.replace("marker=dict(color='#FF453A', size=9, symbol='circle-x', line=dict(width=1, color='white')),", 
                          "marker=dict(color='#FF453A', size=12, symbol='circle', line=dict(width=2, color='white')), text=['🔴']*len(peaks), textposition='top center', textfont=dict(size=16),")

# Edit 8, 9, 10: Extra cards in About section
old_about = "    # ----------------- ABOUT THE PREDICTION -----------------"
new_about = """    # ----------------- ABOUT THE PREDICTION -----------------
    st.markdown('<div class="neon-subheader">ℹ️ Methodology & Dataset</div>', unsafe_allow_html=True)
    col_m1, col_m2, col_m3 = st.columns(3)
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
            <div style="color: #A0A5C0; font-family: monospace; font-size: 0.8rem;">
            CSV<br>↓<br>Normalization<br>↓<br>XGBoost<br>↓<br>Peak Detection<br>↓<br>Transit Candidate
            </div>
        </div>''', unsafe_allow_html=True)
    with col_m3:
        st.markdown('''<div class="detail-box" style="text-align: center;">
            <h4>⚠️ Current Limitations</h4>
            <div style="color: #A0A5C0; font-size: 0.9rem; text-align: left; padding-left: 10px;">
            • Binary classification only<br>
            • Uses Kepler training dataset<br>
            • Orbital period requires timestamps<br>
            • Multi-class planned
            </div>
        </div>''', unsafe_allow_html=True)

    st.markdown('<div class="neon-subheader">ℹ️ Why This Prediction?</div>', unsafe_allow_html=True)"""
content = content.replace(old_about, new_about)

# Edit 11: Noise reduction card
confidence_checklist = """            <ul style="list-style-type: none; padding-left: 0;">
                <li>✓ Flux normalized</li>
                <li>✓ Noise reduced</li>"""
new_confidence_checklist = """            <ul style="list-style-type: none; padding-left: 0;">
                <li>✓ <b>Noise Reduction:</b> StandardScaler Applied</li>
                <li>✓ Flux normalized</li>"""
content = content.replace(confidence_checklist, new_confidence_checklist)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
