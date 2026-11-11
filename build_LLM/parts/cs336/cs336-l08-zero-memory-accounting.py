def zero_memory_accounting(parameter_bytes: int, gradient_bytes: int, optimizer_bytes: int, world_size: int) -> dict:
    """
    Returns a dict of integer byte counts: ddp, zero1, zero2, zero3.
    """
    def ceil_div(numerator: int, denominator: int) -> int:
        return (numerator + denominator - 1) // denominator

    ddp = parameter_bytes + gradient_bytes + optimizer_bytes
    zero1 = parameter_bytes + gradient_bytes + ceil_div(optimizer_bytes, world_size)
    zero2 = parameter_bytes + ceil_div(gradient_bytes + optimizer_bytes, world_size)
    zero3 = ceil_div(parameter_bytes + gradient_bytes + optimizer_bytes, world_size)

    return {
        "ddp": ddp,
        "zero1": zero1,
        "zero2": zero2,
        "zero3": zero3,
    }
