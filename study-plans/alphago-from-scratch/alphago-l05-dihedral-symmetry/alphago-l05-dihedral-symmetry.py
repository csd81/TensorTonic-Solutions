import numpy as np

def apply_go_symmetry(
    state_planes: np.ndarray,
    policy: np.ndarray,
    value: float,
    transform_id: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns transformed state and policy arrays, followed by the unchanged value.
    """
    # 1. Parse the board size from the state planes
    _, n, _ = state_planes.shape
    
    # 2. Extract and reshape the board portion of the policy
    board_policy = policy[:-1].reshape(n, n)
    pass_policy = policy[-1:]
    
    # 3. Determine if we need to apply a left-right reflection
    reflect = transform_id >= 4
    rot_k = transform_id % 4  # Number of 90-degree counterclockwise rotations
    
    # 4. Transform state planes (axis -1 flips columns horizontally)
    transformed_state = state_planes.copy()
    if reflect:
        transformed_state = np.flip(transformed_state, axis=-1)
    if rot_k > 0:
        transformed_state = np.rot90(transformed_state, k=rot_k, axes=(-2, -1))
        
    # 5. Transform the board policy grid identically
    transformed_policy_grid = board_policy.copy()
    if reflect:
        transformed_policy_grid = np.flip(transformed_policy_grid, axis=-1)
    if rot_k > 0:
        transformed_policy_grid = np.rot90(transformed_policy_grid, k=rot_k, axes=(-2, -1))
        
    # 6. Flatten the grid, append the pass token, and ensure memory is newly allocated
    transformed_policy = np.concatenate([transformed_policy_grid.ravel(), pass_policy])
    
    # Ensure explicit new allocation and dtype matching
    return (
        np.array(transformed_state, dtype=state_planes.dtype, copy=True),
        np.array(transformed_policy, dtype=policy.dtype, copy=True),
        value
    )
