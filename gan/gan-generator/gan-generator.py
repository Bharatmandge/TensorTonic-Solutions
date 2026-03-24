import numpy as np

def generator(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Generate fake data from noise vectors.

    Args:
        z: Input noise of shape (batch_size, noise_dim)
        output_dim: Dimension of output data

    Returns:
        fake_data: Generated data of shape (batch_size, output_dim)
    """
    batch_size, noise_dim = z.shape

    # Initialize weights and bias
    W = np.random.randn(noise_dim, output_dim) * 0.01
    b = np.zeros((1, output_dim))

    # Linear transformation
    x_hat = np.dot(z, W) + b

    # Activation (Tanh)
    fake_data = np.tanh(x_hat)

    return fake_data