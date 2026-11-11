import numpy as np

def vae_forward(x: list, W_enc: list, b_enc: list, W_mu: list, b_mu: list, W_logvar: list, b_logvar: list, W_dec: list, b_dec: list, z_sample: list) -> dict:
    """
    Returns hidden, mu, log_var, z, and reconstruction as lists.
    """
    # Convert all inputs to NumPy float64 arrays
    x = np.array(x, dtype=np.float64)
    W_enc = np.array(W_enc, dtype=np.float64)
    b_enc = np.array(b_enc, dtype=np.float64)
    W_mu = np.array(W_mu, dtype=np.float64)
    b_mu = np.array(b_mu, dtype=np.float64)
    W_logvar = np.array(W_logvar, dtype=np.float64)
    b_logvar = np.array(b_logvar, dtype=np.float64)
    W_dec = np.array(W_dec, dtype=np.float64)
    b_dec = np.array(b_dec, dtype=np.float64)
    epsilon = np.array(z_sample, dtype=np.float64)
    
    # 1. Encoder Forward Pass: h = ReLU(W_e * x + b_e)
    h = np.maximum(0.0, np.dot(W_enc, x) + b_enc)
    
    # 2. Latent Projections: mu and log_var
    mu = np.dot(W_mu, h) + b_mu
    log_var = np.dot(W_logvar, h) + b_logvar
    
    # 3. Reparameterization Trick: z = mu + exp(log_var / 2) * epsilon
    z = mu + np.exp(log_var / 2.0) * epsilon
    
    # 4. Decoder Forward Pass: r = sigmoid(W_d * z + b_d)
    r = 1.0 / (1.0 + np.exp(-(np.dot(W_dec, z) + b_dec)))
    
    # Round results to 4 decimal places and convert to native Python lists
    return {
        "hidden": np.round(h, 4).tolist(),
        "mu": np.round(mu, 4).tolist(),
        "log_var": np.round(log_var, 4).tolist(),
        "z": np.round(z, 4).tolist(),
        "reconstruction": np.round(r, 4).tolist()
    }
