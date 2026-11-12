import math


def rotate_image(image: list, angle_degrees: float) -> list:
  """Returns the counterclockwise nearest-neighbor rotation."""
  H = len(image)
  W = len(image[0])

  cy = (H - 1) / 2.0
  cx = (W - 1) / 2.0

  theta = math.radians(angle_degrees)
  cos_t = math.cos(theta)
  sin_t = math.sin(theta)

  output = [[0] * W for _ in range(H)]

  for i in range(H):
    dy = i - cy
    for j in range(W):
      dx = j - cx

      sy = round(cy + dy * cos_t + dx * sin_t)
      sx = round(cx - dy * sin_t + dx * cos_t)

      if 0 <= sy < H and 0 <= sx < W:
        output[i][j] = image[sy][sx]

  return output
