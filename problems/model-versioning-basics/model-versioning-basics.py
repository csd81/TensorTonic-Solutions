def promote_model(models: list) -> str:
    """
    Returns the model name as a string.
    """
    best_model = max(
        models,
        key=lambda m: (m["accuracy"], -m["latency"], m["timestamp"])
    )
    return best_model["name"]
