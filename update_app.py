import re
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

start_idx = content.find("    # Show ground truth if available")

new_content = """    # Show ground truth if available
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
            <div class="metric-label">Prediction</div>
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
        st.metric("Detected Dips", transit_stats['num_transits'])
    with col_t2:
        st.metric("Depth", f"{transit_stats['avg_depth_pct']:.3f}%" if transit_stats['num_transits'] > 0 else "0.000%")
    with col_t3:
        st.metric("Duration", f"{transit_stats['avg_duration']:.1f} bins" if transit_stats['num_transits'] > 0 else "N/A")
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
            st.success("**AI Summary:** The uploaded light curve was normalized and analyzed using an XGBoost classifier followed by peak-based transit detection. Statistically significant transit-like dips were identified. This observation is a strong candidate for an exoplanet transit.")
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
                marker=dict(color='#FF453A', size=9, symbol='circle-x', line=dict(width=1, color='white')),
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
    if len(peaks) > 0:
        st.markdown("#### 🔭 Detected Transit Events")
        for i, peak in enumerate(peaks):
            depth = transit_stats['depths_pct'][i]
            duration = transit_stats['durations'][i]
            snr_val = transit_stats['snr']
            st.markdown(f"**Event {i+1}** - Center: `{peak}` | Depth: `{depth:.3f}%` | Duration: `{duration:.1f} samples` | SNR: `{snr_val:.1f}`")
    else:
        st.markdown("#### 🔭 Detected Transit Events")
        st.write("No statistically significant transit events detected.")
        
    # ----------------- ABOUT THE PREDICTION -----------------
    st.markdown('<div class="neon-subheader">ℹ️ Why This Prediction?</div>', unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown('''
        <div class="detail-box">
            <h4>⚙️ AI Architecture</h4>
            <div style="font-family: monospace; font-size: 0.9rem; color: #A0A5C0; text-align: center; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px;">
                Input: 3197 Flux Values<br>
                ↓<br>
                StandardScaler<br>
                ↓<br>
                XGBoost Classifier<br>
                ↓<br>
                Probability Score<br>
                ↓<br>
                SciPy find_peaks()<br>
                ↓<br>
                Transit Parameters
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
    with col_info2:
        conf_cat = "Strong candidate (>90%)" if confidence > 0.9 else "Moderate candidate (70-90%)" if confidence > 0.7 else "Weak candidate (<70%)"
        st.markdown(f'''
        <div class="detail-box">
            <h4>✅ Confidence Checklist</h4>
            <p>Confidence: <strong>{conf_cat}</strong></p>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>✓ Flux normalized</li>
                <li>✓ Noise reduced</li>
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
"""

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content[:start_idx] + new_content)
