def polynomial_features(values: list, degree: int) -> list:
    """
    Returns powers from zero through degree for every value.
    """
    return [[x**d for d in range(degree + 1)] for x in values]
