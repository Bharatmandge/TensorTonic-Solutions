import numpy as np

def discriminator(x: np.ndarray) -> np.ndarray:
    scores = np.mean(x, axis=1, keepdims=True)
    probs = 1/(1+np.exp(-scores))
    return probs