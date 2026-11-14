def feature_store_lookup(feature_store: dict, requests: list, defaults: dict) -> list:
    """
    Returns a list of feature dictionaries.
    """
    results = []
    for req in requests:
        offline_features = feature_store.get(req["user_id"], defaults)
        combined = {**offline_features, **req["online_features"]}
        results.append(combined)
    return results
