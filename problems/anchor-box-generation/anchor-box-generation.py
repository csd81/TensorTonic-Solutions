import math


def generate_anchors(
    feature_size: int,
    image_size: float,
    scales: list[float],
    aspect_ratios: list[float],
) -> list[list[float]]:
  """Returns a list of [x1, y1, x2, y2] anchor boxes."""
  stride = image_size / feature_size
  anchors = []

  for i in range(feature_size):
    cy = (i + 0.5) * stride
    for j in range(feature_size):
      cx = (j + 0.5) * stride
      for s in scales:
        for r in aspect_ratios:
          sqrt_r = math.sqrt(r)
          w = s * sqrt_r
          h = s / sqrt_r
          anchors.append(
              [float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2)]
          )

  return anchors
