import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> dict:
    """
    Returns hidden_state and cell_state as float64 arrays.
    """
    x = np.concatenate((h_prev, x_t), axis=-1)
    f_t = sigmoid((x @ W_f.T) + b_f)
    i_t = sigmoid((x @ W_i.T) + b_i)
    C_tild = np.tanh((x @ W_c.T) + b_c)
    C_t = f_t * C_prev + i_t * C_tild
    o_t = sigmoid((x @ W_o.T) + b_o)
    h_t = o_t * np.tanh(C_t)

    return {
        "hidden_state": h_t,
        "cell_state": C_t
    }
