import numpy as np

def update_cell_state(C_prev: np.ndarray, f_t: np.ndarray,
                      i_t: np.ndarray, c_tilde: np.ndarray) -> np.ndarray:
    """
    Returns the float64 updated cell state.
    """
    res = np.multiply(f_t, C_prev) + np.multiply(i_t, c_tilde)
    
    return np.asarray(res, dtype=np.float64)