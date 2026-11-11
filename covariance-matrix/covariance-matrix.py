import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    # Write code here
    X = np.array(X, dtype=float)
    c = X - np.mean(X, axis=0)
    return c.T @ c / (X.shape[0]-1)