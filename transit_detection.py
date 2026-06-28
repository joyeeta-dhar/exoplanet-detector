import numpy as np
from scipy.signal import find_peaks

def detect_transits(normalized_flux, prominence=1.0, distance=30, height=None):
    """
    Detect exoplanet transit dips using SciPy's find_peaks on the inverted normalized signal.
    
    Args:
        normalized_flux (np.ndarray): 1D array of normalized flux values.
        prominence (float): Prominence threshold for peak detection.
        distance (int): Minimum sample distance between consecutive peaks.
        height (float or None): Minimum height of the peaks in the inverted signal.
        
    Returns:
        peaks (np.ndarray): Indices of detected peaks (transit centers).
        properties (dict): Properties of detected peaks, including:
            - 'prominences'
            - 'widths'
            - 'left_ips' (left interpolation points for widths)
            - 'right_ips' (right interpolation points for widths)
            - 'peak_heights' (height of the peaks)
    """
    # Invert the signal because transits are dips (negative peaks)
    inverted_signal = -normalized_flux
    
    # Run peak detection
    peaks, properties = find_peaks(
        inverted_signal,
        prominence=prominence,
        distance=distance,
        height=height,
        width=True  # Required to calculate width (duration) properties
    )
    
    # Add inverted signal for visualization/reference
    properties['inverted_signal'] = inverted_signal
    
    return peaks, properties
