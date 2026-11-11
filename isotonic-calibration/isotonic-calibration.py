from bisect import bisect_right


def calibrate_isotonic(cal_labels: list, cal_probs: list, new_probs: list) -> list:
    """
    Returns a list of calibrated probabilities.
    """
    # 1. Sort calibration pairs by ascending probability
    pairs = sorted(zip(cal_probs, cal_labels), key=lambda x: x[0])
    sorted_probs = [p for p, _ in pairs]
    sorted_labels = [y for _, y in pairs]

    # 2. Pool Adjacent Violators (PAV) using a block stack
    # Each block is stored as [sum_of_labels, count_of_points]
    blocks = []
    for y in sorted_labels:
        blocks.append([float(y), 1])
        while len(blocks) >= 2:
            prev_mean = blocks[-2][0] / blocks[-2][1]
            curr_mean = blocks[-1][0] / blocks[-1][1]
            if prev_mean > curr_mean:
                last_sum, last_cnt = blocks.pop()
                blocks[-1][0] += last_sum
                blocks[-1][1] += last_cnt
            else:
                break

    # Expand blocks back into per-point fitted values c_i
    fitted = []
    for block_sum, count in blocks:
        mean_val = block_sum / count
        fitted.extend([mean_val] * count)

    # 3. Clamp outside range and linearly interpolate inside range
    calibrated = []
    for p in new_probs:
        if p <= sorted_probs[0]:
            calibrated.append(float(fitted[0]))
        elif p >= sorted_probs[-1]:
            calibrated.append(float(fitted[-1]))
        else:
            # Find index idx such that sorted_probs[idx - 1] <= p < sorted_probs[idx]
            idx = bisect_right(sorted_probs, p)
            x0, x1 = sorted_probs[idx - 1], sorted_probs[idx]
            y0, y1 = fitted[idx - 1], fitted[idx]
            t = (p - x0) / (x1 - x0)
            calibrated.append(float(y0 + t * (y1 - y0)))

    return calibrated
