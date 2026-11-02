import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    var = np.var(x, ddof=1)
    std = np.std(x, ddof=1)
    res = dict()
    res["variance"] = float(var)
    res["standard_deviation"] = float(std)

    return res