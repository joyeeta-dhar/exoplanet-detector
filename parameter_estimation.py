import numpy as np

def estimate_transit_parameters(raw_flux, normalized_flux, peaks, properties):
    """
    Estimate transit depth, duration (in sample units), and signal-to-noise ratio (SNR).
    
    Args:
        raw_flux (np.ndarray): 1D array of original raw flux values.
        normalized_flux (np.ndarray): 1D array of standardized normalized flux values.
        peaks (np.ndarray): Detected peak indices (transit centers).
        properties (dict): Properties dict returned by detect_transits containing 'widths' and width bounds.
        
    Returns:
        results (dict): Dictionary of estimated parameters:
            - 'num_transits': Number of detected transits.
            - 'depths_raw': List of raw depths (in raw flux units).
            - 'depths_pct': List of relative depths (as % of baseline).
            - 'depths_normalized': List of normalized depths (z-score standard deviations).
            - 'durations': List of durations (in sample units).
            - 'snr': Estimated Signal-to-Noise Ratio.
            - 'avg_depth_pct': Average depth percentage.
            - 'avg_depth_norm': Average normalized depth.
            - 'avg_duration': Average duration in sample units.
            - 'out_of_transit_std': Out-of-transit noise standard deviation.
            - 'transit_mask': Boolean mask indicating transit regions (True = transit).
    """
    n = len(raw_flux)
    
    # If no transits are detected, return default empty/zero values
    if len(peaks) == 0:
        overall_noise = np.std(raw_flux)
        return {
            'num_transits': 0,
            'depths_raw': [],
            'depths_pct': [],
            'depths_normalized': [],
            'durations': [],
            'snr': 0.0,
            'avg_depth_pct': 0.0,
            'avg_depth_norm': 0.0,
            'avg_duration': 0.0,
            'out_of_transit_std': overall_noise if overall_noise > 0 else 1.0,
            'transit_mask': np.zeros(n, dtype=bool)
        }
        
    # 1. Create a mask for transit regions to calculate out-of-transit noise
    transit_mask = np.zeros(n, dtype=bool)
    widths = properties.get('widths', np.ones_like(peaks) * 10)
    
    for peak, width in zip(peaks, widths):
        half_w = int(np.ceil(width * 0.75))  # Mask 1.5x the width (0.75 on each side)
        start = max(0, peak - half_w)
        end = min(n, peak + half_w + 1)
        transit_mask[start:end] = True
        
    # Out-of-transit points
    out_of_transit_flux_raw = raw_flux[~transit_mask]
    out_of_transit_flux_norm = normalized_flux[~transit_mask]
    
    # Fallback to entire signal if mask covers everything
    if len(out_of_transit_flux_raw) < 10:
        out_of_transit_flux_raw = raw_flux
        out_of_transit_flux_norm = normalized_flux
        
    # Calculate out-of-transit baseline
    baseline_raw = np.median(out_of_transit_flux_raw)
    baseline_norm = np.median(out_of_transit_flux_norm)
    
    # If the raw flux is centered (contains negative values), we shift it to make it positive
    # for the purpose of calculating a meaningful relative percentage depth.
    is_centered = np.min(raw_flux) < 0
    if is_centered:
        # Shift so the minimum value is 1000.0 to avoid division by zero and preserve relative proportions
        shift_val = -np.min(raw_flux) + 1000.0
        shifted_raw_flux = raw_flux + shift_val
        shifted_out_of_transit = out_of_transit_flux_raw + shift_val
        baseline_for_pct = np.median(shifted_out_of_transit)
    else:
        shifted_raw_flux = raw_flux
        baseline_for_pct = baseline_raw
        
    # Calculate noise level (std of out-of-transit raw flux)
    noise_raw = np.std(out_of_transit_flux_raw)
    if noise_raw == 0:
        noise_raw = 1e-6
        
    # 2. Calculate depth and duration for each transit
    depths_raw = []
    depths_pct = []
    depths_normalized = []
    durations = []
    
    for i, peak in enumerate(peaks):
        # Raw Depth
        flux_val_raw = raw_flux[peak]
        d_raw = max(0.0, baseline_raw - flux_val_raw)
        depths_raw.append(d_raw)
        
        # Relative Depth (%) - calculated on shifted positive flux if centered
        flux_val_shifted = shifted_raw_flux[peak]
        d_shifted = max(0.0, baseline_for_pct - flux_val_shifted)
        d_pct = (d_shifted / baseline_for_pct) * 100 if baseline_for_pct != 0 else 0.0
        depths_pct.append(d_pct)
        
        # Normalized Depth
        flux_val_norm = normalized_flux[peak]
        d_norm = max(0.0, baseline_norm - flux_val_norm)
        depths_normalized.append(d_norm)
        
        # Duration
        durations.append(widths[i])
        
    # 3. Calculate Signal-to-Noise Ratio (SNR)
    noise_norm = np.std(out_of_transit_flux_norm)
    if noise_norm == 0:
        noise_norm = 1e-6
    avg_depth_norm = np.mean(depths_normalized)
    snr = avg_depth_norm / noise_norm
    
    return {
        'num_transits': len(peaks),
        'depths_raw': depths_raw,
        'depths_pct': depths_pct,
        'depths_normalized': depths_normalized,
        'durations': durations,
        'snr': float(snr),
        'avg_depth_pct': float(np.mean(depths_pct)),
        'avg_depth_norm': float(avg_depth_norm),
        'avg_duration': float(np.mean(durations)),
        'out_of_transit_std': float(noise_raw),
        'transit_mask': transit_mask
    }
