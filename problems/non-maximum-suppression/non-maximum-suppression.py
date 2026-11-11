def nms(boxes: list, scores: list, iou_threshold: float) -> list:
    if not boxes:
        return []

    def compute_iou(box_a: list, box_b: list) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union_area = area_a + area_b - inter_area

        return inter_area / union_area if union_area > 0.0 else 0.0

    n = len(boxes)
    order = sorted(range(n), key=lambda idx: (-scores[idx], idx))
    suppressed = [False] * n
    selected = []

    for i in range(n):
        curr = order[i]
        if suppressed[curr]:
            continue
        selected.append(curr)
        for j in range(i + 1, n):
            cand = order[j]
            if not suppressed[cand] and compute_iou(boxes[curr], boxes[cand]) >= iou_threshold:
                suppressed[cand] = True

    return selected
