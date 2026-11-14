def image_histogram(image: list) -> list:
  """Returns a list of intensity and count pairs."""
  hist = [0] * 256

  for row in image:
    for pixel in row:
      hist[pixel] += 1

  return [[intensity, count] for intensity, count in enumerate(hist) if count > 0]
