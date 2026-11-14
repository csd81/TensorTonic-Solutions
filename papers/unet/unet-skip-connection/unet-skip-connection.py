import numpy as np

def crop_and_concat(encoder_features: np.ndarray,
                    decoder_features: np.ndarray) -> np.ndarray:
    """
    Returns the centered encoder crop concatenated with decoder features.
    """
    encoder_features = encoder_features.astype(np.float64, copy=False)
    decoder_features = decoder_features.astype(np.float64, copy=False)

    _, enc_h, enc_w, _ = encoder_features.shape
    _, dec_h, dec_w, _ = decoder_features.shape

    # Integer division naturally discards the extra row/column from the bottom/right when odd
    start_h = (enc_h - dec_h) // 2
    start_w = (enc_w - dec_w) // 2

    cropped_encoder = encoder_features[
        :,
        start_h:start_h + dec_h,
        start_w:start_w + dec_w,
        :
    ]

    # Concatenate encoder channels before decoder channels along the last axis
    return np.concatenate([cropped_encoder, decoder_features], axis=-1).astype(np.float64, copy=False)
