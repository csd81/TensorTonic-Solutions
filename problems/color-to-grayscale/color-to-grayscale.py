def color_to_grayscale(image: list) -> list:
    """
    Returns the luminance value of every RGB pixel.
    """
    return [
        [
            round(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2], 6)
            for pixel in row
        ]
        for row in image
    ]
