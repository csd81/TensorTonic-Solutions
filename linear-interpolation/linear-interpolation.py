def linear_interpolation(values: list) -> list:
    """
    Returns a copy with every missing value interpolated.
    """
    result = list(values)
    left = 0

    for right in range(1, len(result)):
        if result[right] is not None:
            # Check if there is a gap of None values between left and right
            if right - left > 1:
                v_left = result[left]
                v_right = result[right]
                span = right - left

                for j in range(left + 1, right):
                    result[j] = v_left + ((j - left) / span) * (v_right - v_left)

            # Update left pointer to the current known value
            left = right

    return result
