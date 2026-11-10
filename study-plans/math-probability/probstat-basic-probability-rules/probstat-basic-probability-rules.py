def basic_probability(p_a: float, p_b: float, p_a_and_b: float) -> list:
    """
    Returns four rounded probability values in the required order.
    """
    union = p_a + p_b - p_a_and_b
    comp_a = 1.0 - p_a
    comp_b = 1.0 - p_b
    a_outside_b = p_a - p_a_and_b

    return[
        round(union, 4),
        round(comp_a, 4),
        round(comp_b, 4),
        round(a_outside_b, 4)
    ]
        