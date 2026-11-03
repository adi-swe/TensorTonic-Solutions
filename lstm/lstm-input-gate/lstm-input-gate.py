import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> dict:
    """
    Returns input_gate and candidate_state as float64 arrays.
    """
    
    x = np.concatenate((h_prev, x_t), axis=-1)

    res_i = sigmoid(x @ W_i.T + b_i)
    res_c = np.tanh(x @ W_c.T + b_c)

    return {
        "input_gate": np.asarray(res_i, dtype=np.float64),
        "candidate_state": np.asarray(res_c, dtype=np.float64)
    }