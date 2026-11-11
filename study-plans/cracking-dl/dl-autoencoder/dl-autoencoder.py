import numpy as np

def autoencoder(x: list, W_enc: list, b_enc: list, W_dec: list, b_dec: list) -> dict:
    """
    Returns a dictionary of lists: encoded, decoded.
    """
    # Convert all inputs to NumPy arrays with float64 precision
    x_arr = np.array(x, dtype=np.float64)
    W_enc_arr = np.array(W_enc, dtype=np.float64)
    b_enc_arr = np.array(b_enc, dtype=np.float64)
    W_dec_arr = np.array(W_dec, dtype=np.float64)
    b_dec_arr = np.array(b_dec, dtype=np.float64)
    
    # Encoder Forward Pass: z = ReLU(W_enc * x + b_enc)
    z_linear = np.dot(W_enc_arr, x_arr) + b_enc_arr
    z = np.maximum(0, z_linear)
    
    # Decoder Forward Pass: x_hat = Sigmoid(W_dec * z + b_dec)
    x_hat_linear = np.dot(W_dec_arr, z) + b_dec_arr
    x_hat = 1 / (1 + np.exp(-x_hat_linear))
    
    # Round results to 4 decimal places and convert back to Python lists
    encoded_rounded = np.round(z, 4).tolist()
    decoded_rounded = np.round(x_hat, 4).tolist()
    
    return {
        'encoded': encoded_rounded,
        'decoded': decoded_rounded
    }
