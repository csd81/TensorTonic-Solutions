import math

def evaluate_shadow(production_log: list, shadow_log: list, criteria: dict) -> dict:
    """
    Returns a dictionary with the promotion decision and metrics.
    """
    n = len(production_log)
    prod_correct = 0
    shadow_correct = 0
    agreements = 0
    shadow_latencies = []

    for prod, shad in zip(production_log, shadow_log):
        if prod["prediction"] == prod["actual"]:
            prod_correct += 1
        if shad["prediction"] == shad["actual"]:
            shadow_correct += 1
        if prod["prediction"] == shad["prediction"]:
            agreements += 1
        shadow_latencies.append(shad["latency_ms"])

    production_accuracy = prod_correct / n
    shadow_accuracy = shadow_correct / n
    accuracy_gain = shadow_accuracy - production_accuracy
    agreement_rate = agreements / n

    # Nearest-rank P95 latency
    shadow_latencies.sort()
    p95_index = math.ceil(0.95 * n) - 1
    shadow_latency_p95 = shadow_latencies[p95_index]

    # Check all three promotion criteria
    promote = bool(
        accuracy_gain >= criteria["min_accuracy_gain"]
        and shadow_latency_p95 <= criteria["max_latency_p95"]
        and agreement_rate >= criteria["min_agreement_rate"]
    )

    return {
        "promote": promote,
        "metrics": {
            "shadow_accuracy": shadow_accuracy,
            "production_accuracy": production_accuracy,
            "accuracy_gain": accuracy_gain,
            "shadow_latency_p95": shadow_latency_p95,
            "agreement_rate": agreement_rate,
        },
    }
