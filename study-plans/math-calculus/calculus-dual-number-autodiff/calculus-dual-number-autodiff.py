import math

class DualNumber:
    def __init__(self, val: float, deriv: float):
        self.val = val
        self.deriv = deriv

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return DualNumber(self.val + other, self.deriv)
        return DualNumber(self.val + other.val, self.deriv + other.deriv)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return DualNumber(self.val * other, self.deriv * other)
        return DualNumber(self.val * other.val, self.val * other.deriv + self.deriv * other.val)

    def __rmul__(self, other):
        return self.__mul__(other)

    def sin(self):
        return DualNumber(math.sin(self.val), self.deriv * math.cos(self.val))

    def exp(self):
        exp_val = math.exp(self.val)
        return DualNumber(exp_val, self.deriv * exp_val)


def dual_number_autodiff(x_val: float) -> dict:
    """
    Returns the function value, two derivatives, and their comparison.
    """
    # 1. Dual approach
    x = DualNumber(x_val, 1.0)
    # f(x) = x^2 * sin(x) + e^x
    f_dual = (x * x) * x.sin() + x.exp()
    
    # 2. Analytical approach
    f_deriv_analytical = 2 * x_val * math.sin(x_val) + (x_val ** 2) * math.cos(x_val) + math.exp(x_val)
    
    # 3. Precision comparison
    match = math.isclose(f_dual.deriv, f_deriv_analytical, abs_tol=1e-10)
    
    return {
        "f_deriv_analytical": f_deriv_analytical,
        "f_deriv_dual": f_dual.deriv,
        "f_val": f_dual.val,
        "match": match
    }
