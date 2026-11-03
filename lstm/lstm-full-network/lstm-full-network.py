
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def lstm_forward(X: np.ndarray, W_f: np.ndarray, W_i: np.ndarray,
                 W_c: np.ndarray, W_o: np.ndarray, b_f: np.ndarray,
                 b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray,
                 W_y: np.ndarray, b_y: np.ndarray) -> dict:
    """
    Returns outputs, final_hidden_state, and final_cell_state.
    All returned arrays use float64.
    """
    X = np.asarray(X, dtype=np.float64)

    N, T, D = X.shape
    H = W_f.shape[0]
    O = W_y.shape[0]

    h_prev = np.zeros((N, H), dtype=np.float64)
    C_prev = np.zeros((N, H), dtype=np.float64)

    outputs = np.zeros((N, T, O), dtype=np.float64)

    for t in range(T):
        x_t = X[:, t, :]

        x = np.concatenate((h_prev, x_t), axis=-1)

        f_t = sigmoid(x @ W_f.T + b_f)
        i_t = sigmoid(x @ W_i.T + b_i)
        C_tild = np.tanh(x @ W_c.T + b_c)

        C_t = f_t * C_prev + i_t * C_tild

        o_t = sigmoid(x @ W_o.T + b_o)
        h_t = o_t * np.tanh(C_t)

        y_t = h_t @ W_y.T + b_y

        outputs[:, t, :] = y_t

        h_prev = h_t
        C_prev = C_t

    return {
        "outputs": outputs,
        "final_hidden_state": np.asarray(h_prev, dtype=np.float64),
        "final_cell_state": np.asarray(C_prev, dtype=np.float64)
    }
