from collections import Counter
import numpy as np
from scipy import stats

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    mean = np.mean(x)
    median = np.median(x)
    mode = stats.mode(x).mode
    res = dict()
    res["mean"] = float(mean)
    res["median"] = float(median)
    res["mode"] = float(mode)
    

    return res