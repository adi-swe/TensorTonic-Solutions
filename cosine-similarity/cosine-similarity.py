import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    x = np.dot(a,b)
    y = np.linalg.norm(a) * np.linalg.norm(b)
    if y == 0:
        return 0.0
    return float(x/y)