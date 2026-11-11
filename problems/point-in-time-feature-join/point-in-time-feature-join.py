import bisect
from collections import defaultdict

def point_in_time_join(events: list, feature_rows: list, ttl: int) -> list:
    """
    Returns one dictionary per event with the feature value that was known at that time.
    """
    # Group feature rows by entity_id
    features_by_entity = defaultdict(list)
    for row in feature_rows:
        features_by_entity[row["entity_id"]].append((row["timestamp"], row["value"]))

    # Sort each entity's feature rows by timestamp ascending
    # Store separate lists of timestamps and values for fast bisect lookups
    sorted_features = {}
    for entity_id, rows in features_by_entity.items():
        rows.sort(key=lambda x: x[0])
        timestamps = [r[0] for r in rows]
        values = [r[1] for r in rows]
        sorted_features[entity_id] = (timestamps, values)

    result = []
    for event in events:
        entity_id = event["entity_id"]
        event_ts = event["timestamp"]
        chosen_value = None

        if entity_id in sorted_features:
            timestamps, values = sorted_features[entity_id]
            
            # Find the rightmost feature timestamp <= event_ts
            idx = bisect.bisect_right(timestamps, event_ts) - 1
            
            # Check if a valid prior feature exists and is within the ttl window
            if idx >= 0 and (event_ts - timestamps[idx]) <= ttl:
                chosen_value = values[idx]

        result.append({
            "entity_id": entity_id,
            "timestamp": event_ts,
            "value": chosen_value
        })

    return result
