import numpy as np

def bank_conflict_analysis(byte_addresses: np.ndarray, num_banks: int = 32, bank_width_bytes: int = 4) -> dict:
    """
    Returns a dict of int64 arrays: bank_ids and conflict_degree.
    """
    # Compute the bank ID for each lane
    bank_ids = (byte_addresses // bank_width_bytes) % num_banks
    
    # Pair each bank ID with its corresponding address to keep track of uniquely requested addresses
    # Using a structured array or unique pairs to find unique (bank, address) combinations
    # Since we only care about counting unique addresses per bank present in the current batch:
    unique_pairs = np.unique(np.stack((bank_ids, byte_addresses), axis=1), axis=0)
    
    # Count how many unique addresses exist for each bank
    unique_banks, counts = np.unique(unique_pairs[:, 0], return_counts=True)
    
    # Map the counts back to the original lane order using bank IDs
    bank_to_count = np.zeros(num_banks, dtype=np.int64)
    bank_to_count[unique_banks] = counts
    conflict_degrees = bank_to_count[bank_ids]
    
    return {
        "bank_ids": bank_ids.astype(np.int64),
        "conflict_degree": conflict_degrees.astype(np.int64)
    }
