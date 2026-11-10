import torch

def encode_go_history(history: list[torch.Tensor], to_play: int) -> torch.Tensor:
    """
    Encodes the current board and up to seven earlier boards into a 17-channel float32 
    tensor from the perspective of the current player (`to_play`).
    
    Returns:
        A torch.Tensor of shape (17, board_size, board_size) on the input history device.
    """
    # 1. Gather properties from the newest state
    device = history[0].device
    board_size = history[0].shape[-1]
    
    # 2. Pre-allocate the full output feature map
    # 16 planes for 8 history steps (X and Y) + 1 plane for the color to play
    encoded = torch.zeros((17, board_size, board_size), dtype=torch.float32, device=device)
    
    # 3. Populate feature planes for available history steps
    # Up to 8 steps are considered (each providing 2 planes)
    num_steps = min(len(history), 8)
    for t in range(num_steps):
        board = history[t]
        
        # Current player's stones (X_t)
        encoded[2 * t] = (board == to_play).to(torch.float32)
        
        # Opponent's stones (Y_t)
        encoded[2 * t + 1] = (board == -to_play).to(torch.float32)
        
    # 4. Populate the final 17th plane (C) indicating color to play
    # 1.0 if Black (1) is to play, 0.0 if White (-1) is to play
    if to_play == 1:
        encoded[16] = 1.0
        
    return encoded
