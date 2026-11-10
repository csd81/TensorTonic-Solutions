import numpy as np

def seq2seq_forward(src: list, tgt: list, W_enc_embed: list, W_dec_embed: list, W_enc: list, b_enc: list, W_dec: list, b_dec: list, W_out: list, b_out: list) -> dict:
    """
    Returns a dictionary of lists: encoder_hiddens, decoder_hiddens, logits.
    """
    # Convert all inputs to NumPy arrays with float64 precision
    W_enc_embed = np.array(W_enc_embed, dtype=np.float64)
    W_dec_embed = np.array(W_dec_embed, dtype=np.float64)
    W_enc = np.array(W_enc, dtype=np.float64)
    b_enc = np.array(b_enc, dtype=np.float64)
    W_dec = np.array(W_dec, dtype=np.float64)
    b_dec = np.array(b_dec, dtype=np.float64)
    W_out = np.array(W_out, dtype=np.float64)
    b_out = np.array(b_out, dtype=np.float64)

    # Initialize the encoder hidden state to zeros
    H = len(b_enc)
    h = np.zeros(H, dtype=np.float64)
    
    # Encoder forward pass
    encoder_hiddens = []
    for token in src:
        e = W_enc_embed[token]
        concat = np.concatenate([h, e])
        h = np.tanh(np.dot(W_enc, concat) + b_enc)
        encoder_hiddens.append(h.copy())
        
    # Decoder forward pass (initialized with the final encoder hidden state)
    decoder_hiddens = []
    logits = []
    for token in tgt:
        e = W_dec_embed[token]
        concat = np.concatenate([h, e])
        h = np.tanh(np.dot(W_dec, concat) + b_dec)
        decoder_hiddens.append(h.copy())
        
        # Compute logits for the current step
        l = np.dot(W_out, h) + b_out
        logits.append(l)
        
    # Round final results to four decimal places and convert to nested lists
    return {
        'encoder_hiddens': np.round(encoder_hiddens, 4).tolist(),
        'decoder_hiddens': np.round(decoder_hiddens, 4).tolist(),
        'logits': np.round(logits, 4).tolist()
    }
