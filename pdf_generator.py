from fpdf import FPDF
import io

def generate_pdf_report(pred_label, confidence, transit_stats):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(200, 10, txt="AI Exoplanet Detection Mission Report", ln=True, align='C')
    
    pdf.set_font("Helvetica", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Mission Summary:", ln=True)
    pdf.set_font("Helvetica", style="I", size=11)
    
    # Generate summary string
    if pred_label == "Potential Transit Candidate":
        summary = (f"The uploaded light curve contains {transit_stats['num_transits']} statistically "
                   f"significant transit-like dips. This light curve is a strong candidate for "
                   f"follow-up astronomical analysis.")
    else:
        summary = (f"The uploaded light curve does not show strong evidence of exoplanet transits. "
                   f"Detected dips ({transit_stats['num_transits']}) are likely due to stellar noise "
                   f"or instrumental artifacts.")
                   
    pdf.multi_cell(0, 10, txt=summary)
    
    pdf.ln(10)
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(200, 10, txt="Metrics Details:", ln=True)
    
    pdf.set_font("Helvetica", size=11)
    pdf.cell(200, 10, txt=f"- AI Decision: {pred_label}", ln=True)
    pdf.cell(200, 10, txt=f"- Model Confidence: {confidence*100:.2f}%", ln=True)
    
    if transit_stats['num_transits'] > 0:
        pdf.cell(200, 10, txt=f"- Estimated Transit Depth: {transit_stats['avg_depth_pct']:.3f}%", ln=True)
        pdf.cell(200, 10, txt=f"- Estimated Duration: {transit_stats['avg_duration']:.1f} samples", ln=True)
        pdf.cell(200, 10, txt=f"- Signal-to-Noise Ratio (SNR): {transit_stats['snr']:.2f}", ln=True)
    else:
        pdf.cell(200, 10, txt="- Estimated Transit Depth: N/A", ln=True)
        pdf.cell(200, 10, txt="- Estimated Duration: N/A", ln=True)
        pdf.cell(200, 10, txt="- Signal-to-Noise Ratio (SNR): N/A", ln=True)
        
    pdf.ln(10)
    pdf.cell(200, 10, txt="- Orbital Period: Estimated in sample units; physical period requires timestamped observations.", ln=True)
    
    return bytes(pdf.output(dest='S'))
