import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    # Write code here
    # x = np.array(x, dtype=float)
    pmf = [p if a == 1 else (1 - p) for a in x]
    pmf = np.array(pmf, dtype=float)
    mean = float(p)
    var = float(p * (1-p))

    return {
        "pmf": pmf,
        "mean": mean,
        "variance": var
    }
    