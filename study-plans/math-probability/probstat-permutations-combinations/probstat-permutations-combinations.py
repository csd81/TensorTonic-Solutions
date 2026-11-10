def perms_and_combs(n: int, r: int) -> list:
    """
    Returns permutations, combinations, and n factorial as integers.
    """
    def fac(k: int) -> int:
        f = 1
        for i in range(1, k+1):
            f *= i
        return f
    
    n_f = fac(n)
    r_f = fac(r)
    n_minus_r_f = fac(n-r)

    permutations = n_f // n_minus_r_f
    combinations = n_f // (r_f * n_minus_r_f)
    
    return [permutations, combinations, n_f]