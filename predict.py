import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_resource
def load_scaler(path="models/scaler.pkl"):
    """
    Load the pre-trained StandardScaler using joblib (since joblib was used to save it).
    """
    return joblib.load(path)

@st.cache_resource
def load_model(path="models/xgboost_model.pkl"):
    """
    Load the pre-trained XGBoost model using pickle.
    """
    with open(path, "rb") as f:
        return pickle.load(f)

def load_and_preprocess_csv(file_obj):
    """
    Load and preprocess the uploaded CSV file containing flux values.
    Handles different shapes:
    - 3197 rows, 1 column (vertical time series)
    - 1 row, 3197 columns (horizontal time series)
    - 1 row, 3198 columns (Kepler format with a LABEL column)
    
    Returns:
        flux_2d (np.ndarray): 1x3197 array of flux values.
        true_label (int or None): True label if Kepler format with LABEL is detected.
    """
    try:
        # Read the CSV. We don't assume header or not initially.
        # Let's inspect the first row.
        df = pd.read_csv(file_obj, header=None)
        
        # Check shape
        rows, cols = df.shape
        
        true_label = None
        
        # Case A: 3198 columns (horizontal row, first column is label)
        if cols == 3198:
            # Let's re-read with header=0 to check for "LABEL" or similar headers
            file_obj.seek(0)
            df_header = pd.read_csv(file_obj)
            
            # Check if first column has header containing "label"
            first_col = df_header.columns[0]
            if "label" in str(first_col).lower():
                true_label = int(df_header[first_col].values[0])
                flux_values = df_header.iloc[0, 1:].values.astype(float)
            else:
                # No header, but we assume column 0 is the label
                true_label = int(df.iloc[0, 0])
                flux_values = df.iloc[0, 1:].values.astype(float)
                
        # Case B: 3197 columns (horizontal row, no label)
        elif cols == 3197:
            # We assume it's just one row of flux values
            # Check if there is a header or if the first row is numeric
            try:
                # Try to cast first row to float
                flux_values = df.iloc[0, :].values.astype(float)
            except ValueError:
                # If first row is headers, read again or drop first row
                file_obj.seek(0)
                df_header = pd.read_csv(file_obj)
                flux_values = df_header.iloc[0, :].values.astype(float)
                
        # Case C: 1 column, 3197 rows (vertical flux series)
        elif rows == 3197:
            # Single column of flux values
            try:
                flux_values = df.iloc[:, 0].values.astype(float)
            except ValueError:
                # First row was a header
                file_obj.seek(0)
                df_header = pd.read_csv(file_obj)
                flux_values = df_header.iloc[:, 0].values.astype(float)
                
        # Case D: 1 column, 3198 rows (vertical series with a label at start/end)
        elif rows == 3198:
            # We assume first row is label
            true_label = int(df.iloc[0, 0])
            flux_values = df.iloc[1:, 0].values.astype(float)
            
        else:
            raise ValueError(
                f"Unsupported shape: {rows}x{cols}. The CSV must contain exactly 3197 flux values "
                "(either as a single row/column, or 3198 columns/rows with a label)."
            )
            
        # Ensure length is exactly 3197
        if len(flux_values) != 3197:
            raise ValueError(f"Extracted flux length is {len(flux_values)}, but must be exactly 3197.")
            
        return flux_values.reshape(1, -1), true_label
        
    except Exception as e:
        raise ValueError(f"Error parsing CSV: {str(e)}")

def predict_exoplanet(flux_2d, model, scaler):
    """
    Perform scaling and prediction using the XGBoost model and Scaler.
    
    Args:
        flux_2d (np.ndarray): 1x3197 array of flux values.
        model: Loaded XGBoost classifier.
        scaler: Loaded StandardScaler.
        
    Returns:
        prediction (int): 0 for No Exoplanet, 1 for Exoplanet Detected.
        confidence (float): Confidence score (0.0 to 1.0).
        scaled_flux (np.ndarray): Scaled flux values (1x3197).
    """
    # Align features to StandardScaler's fitted names to avoid warnings
    feature_names = [f"FLUX.{i}" for i in range(1, 3198)]
    df = pd.DataFrame(flux_2d, columns=feature_names)
    
    # Standardize/Scale the flux values
    scaled_flux = scaler.transform(df)
    
    # Run model prediction
    prediction = int(model.predict(scaled_flux)[0])
    probabilities = model.predict_proba(scaled_flux)[0]
    confidence = float(probabilities[prediction])
    
    return prediction, confidence, scaled_flux[0]
