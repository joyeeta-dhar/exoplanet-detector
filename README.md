<div align="center">
  <h1>🌌 AI Exoplanet Detection Dashboard</h1>
  <p><i>An intelligent, space-themed pipeline for detecting exoplanet transits from noisy Kepler stellar light curves.</i></p>
  
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://xgboost.readthedocs.io/"><img src="https://img.shields.io/badge/XGBoost-Machine%20Learning-green" alt="XGBoost"></a>
</div>

## 🔭 Project Overview

Detecting exoplanets involves searching for microscopic dips in starlight caused by planets transiting across their host star. Due to intense stellar noise and instrument artifacts, standard analytical methods often fail. 

This project solves this by using an **XGBoost Machine Learning Classifier** paired with a robust digital signal processing pipeline to isolate, verify, and characterize these exoplanet transits from raw astronomical data.

## ✨ Key Features

- **End-to-End Processing**: Upload a CSV of flux values to immediately view the normalized output and AI verdict.
- **Physical Parameter Estimation**: Automatically calculates `Transit Depth`, `Transit Duration`, and `Signal-to-Noise Ratio (SNR)`.
- **Interactive Visualizations**: Includes Plotly-powered graphs that overlay detected transit dips right on top of the normalized light curves.
- **Automated PDF Reports**: Automatically generates a downloadable mission report with all astronomical findings.
- **SciPy Peak Analysis**: Extracts exact local minima to map out potential exoplanetary orbits.

## ⚙️ AI Architecture & Pipeline

```text
 📄 CSV Input (3,197 Flux Samples)
       ⬇
 ⚖️ StandardScaler Normalization
       ⬇
 🧠 XGBoost ML Classifier
       ⬇
 📊 Probability & Confidence Score
       ⬇
 🔎 SciPy Transit Peak Detection
       ⬇
 📈 Physical Parameter Estimation (Depth, Duration, SNR)
       ⬇
 ✅ Final Astronomical Verdict
```

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/joyeeta-dhar/exoplanet-detector.git
cd exoplanet-detector
```

### 2. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```
*(Dependencies include `streamlit`, `pandas`, `numpy`, `plotly`, `scipy`, `xgboost`, `scikit-learn`, `fpdf`)*

### 3. Launch the Dashboard
```bash
streamlit run app.py
```
Your browser will automatically open the application at `http://localhost:8501`.

## 📡 Dataset
The model is trained and validated on **NASA's Kepler Space Telescope Mission Data** (Public Dataset). 
Each record requires exactly 3,197 relative flux measurements. 

## ⚠️ Limitations & Future Work
- Currently designed for **binary classification** (Exoplanet vs. Noise).
- Physical orbital period cannot be estimated perfectly without raw timestamp parameters.
- Future support planned for detecting **Eclipsing Binaries** and **Stellar Blends**.

---
*Created for astronomical signal analysis and hackathon demonstrations.*
