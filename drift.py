import numpy as np
import pandas as pd

# def calculate_psi(expected, actual, bins=10):
#     breakpoints = np.linspace(0, 100, bins + 1)
#     expected_perc = np.percentile(expected, breakpoints)
    
#     expected_counts = np.histogram(expected, bins=expected_perc)[0]
#     actual_counts = np.histogram(actual, bins=expected_perc)[0]

#     expected_ratio = expected_counts / len(expected)
#     actual_ratio = actual_counts / len(actual)

#     psi = np.sum((actual_ratio - expected_ratio) * 
#                  np.log((actual_ratio + 1e-5) / (expected_ratio + 1e-5)))
    
#     return psi
import numpy as np

def calculate_psi(expected, actual, bins=10):
    expected = np.array(expected)
    actual = np.array(actual)

    # Remove NaNs
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    # If data too small → skip
    if len(expected) < 50 or len(actual) < 50:
        return 0.0

    # Create bins safely
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())

    if min_val == max_val:
        return 0.0  # no variation → no drift

    bin_edges = np.linspace(min_val, max_val, bins + 1)

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_ratio = expected_counts / len(expected)
    actual_ratio = actual_counts / len(actual)

    # Smooth zeros
    expected_ratio = np.clip(expected_ratio, 1e-6, None)
    actual_ratio = np.clip(actual_ratio, 1e-6, None)

    psi = np.sum((actual_ratio - expected_ratio) *
                 np.log(actual_ratio / expected_ratio))

    return psi
def detect_drift(df_old, df_new):
    results = {}
    
    for col in df_old.columns:
        psi = calculate_psi(df_old[col], df_new[col])
        
        if psi > 0.25:
            status = "high drift"
        elif psi > 0.1:
            status = "moderate drift"
        else:
            status = "no drift"
        
        results[col] = {
            "psi": psi,
            "status": status
        }
    
    return results