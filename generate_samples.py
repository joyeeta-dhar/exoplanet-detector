import os
import joblib
import pickle
import numpy as np
import pandas as pd

# Create samples directory
os.makedirs("samples", exist_ok=True)

# Load scaler and model
print("Loading model and scaler to generate verified sample data...")
scaler = joblib.load("models/scaler.pkl")
with open("models/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Helper function to get prediction from scaled signal
def get_prediction_from_scaled(scaled_signal):
    prob = model.predict_proba(scaled_signal.reshape(1, -1))[0]
    pred = model.predict(scaled_signal.reshape(1, -1))[0]
    return pred, prob[pred]

print("Searching for verified Exoplanet and Non-Exoplanet light curves...")

# Generate Non-Exoplanet Sample (Class 0)
np.random.seed(42)
found_non_exo = False
flux_non_exo = None
non_exo_prob = 0.0

for attempt in range(2000):
    # Flat line with some noise
    signal = np.zeros(3197)
    t = np.linspace(0, 100, 3197)
    signal += np.sin(t * np.random.uniform(0.01, 0.1)) * np.random.uniform(0.05, 0.2)
    signal += np.random.normal(0, np.random.uniform(0.01, 0.05), 3197)
    
    # Standardize signal
    signal = (signal - np.mean(signal)) / np.std(signal)
    
    pred, prob = get_prediction_from_scaled(signal)
    if pred == 0 and prob > 0.99:
        flux_non_exo = signal * scaler.scale_ + scaler.mean_
        non_exo_prob = prob
        found_non_exo = True
        print(f"Non-Exoplanet verified: prediction={pred}, confidence={prob:.4f} at attempt {attempt}")
        break

# Generate Exoplanet Sample (Class 1)
np.random.seed(42) # Use the same seed as the successful search
found_exo = False
flux_exo = None
exo_prob = 0.0
best_exo_prob = 0.0
best_exo_signal = None

for attempt in range(10000):
    t = np.linspace(0, 100, 3197)
    signal = np.zeros(3197)
    
    trend_type = np.random.choice(["flat", "sine", "slope"])
    if trend_type == "sine":
        signal += np.sin(t * np.random.uniform(0.01, 0.2)) * np.random.uniform(0.1, 0.5)
    elif trend_type == "slope":
        signal += (t - 50) / 50 * np.random.uniform(-0.5, 0.5)
        
    period = np.random.uniform(10, 45)
    width = np.random.uniform(1, 4)
    depth = np.random.uniform(0.5, 5.0)
    
    num_transits = int(100 / period) + 1
    start_offset = np.random.uniform(0, period)
    for j in range(num_transits):
        center = start_offset + j * period
        mask = (t >= center - width/2) & (t <= center + width/2)
        signal[mask] -= depth
        
        # Add smooth transitions
        ingress = (t >= center - width/2 - 0.3) & (t < center - width/2)
        signal[ingress] -= depth * (t[ingress] - (center - width/2 - 0.3)) / 0.3
        egress = (t > center + width/2) & (t <= center + width/2 + 0.3)
        signal[egress] -= depth * ((center + width/2 + 0.3) - t[egress]) / 0.3

    noise = np.random.normal(0, np.random.uniform(0.01, 0.2), 3197)
    signal += noise
    
    signal = (signal - np.mean(signal)) / np.std(signal)
    
    pred, prob = get_prediction_from_scaled(signal)
    if pred == 1:
        if prob > best_exo_prob:
            best_exo_prob = prob
            best_exo_signal = signal
            print(f"New best exoplanet prediction: prob={prob:.4f} at attempt {attempt}")
            if prob > 0.90:
                flux_exo = signal * scaler.scale_ + scaler.mean_
                exo_prob = prob
                found_exo = True
                print(f"Exoplanet verified: prediction={pred}, confidence={prob:.4f} at attempt {attempt}")
                break

if not found_exo and best_exo_signal is not None:
    print(f"Could not find exoplanet with prob > 0.90. Using best found with prob: {best_exo_prob:.4f}")
    flux_exo = best_exo_signal * scaler.scale_ + scaler.mean_
    exo_prob = best_exo_prob

# Save files
header_cols = ["LABEL"] + [f"FLUX.{i}" for i in range(1, 3198)]

# Format A: Flux values only, no headers
pd.DataFrame({"flux": flux_non_exo}).to_csv("samples/non_exoplanet_flux_only.csv", index=False, header=False)
pd.DataFrame({"flux": flux_exo}).to_csv("samples/exoplanet_flux_only.csv", index=False, header=False)

# Format B: Kepler-style CSV files (header: LABEL, FLUX.1 ... FLUX.3197)
# For exoplanet, LABEL is 2. For non-exoplanet, LABEL is 1.
row_non_exo = [1] + list(flux_non_exo)
pd.DataFrame([row_non_exo], columns=header_cols).to_csv("samples/non_exoplanet_kepler_format.csv", index=False)

row_exo = [2] + list(flux_exo)
pd.DataFrame([row_exo], columns=header_cols).to_csv("samples/exoplanet_kepler_format.csv", index=False)

print("\nSamples generated successfully in samples/ directory:")
print(f"1. samples/non_exoplanet_flux_only.csv (Prob: {non_exo_prob:.4f})")
print(f"2. samples/exoplanet_flux_only.csv (Prob: {exo_prob:.4f})")
print("3. samples/non_exoplanet_kepler_format.csv")
print("4. samples/exoplanet_kepler_format.csv")
