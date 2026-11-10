def rgb_to_grayscale(image: list) -> list:
    """
    Returns the H-by-W grayscale list of floats, rounded to four decimals.
    """
    grayscale_image = []
    
    for row in image:
        new_row = []
        for pixel in row:
            r, g, b = pixel
            # Apply the BT.601 luma formula
            y = 0.299 * r + 0.587 * g + 0.114 * b
            # Round to four decimal places
            new_row.append(round(y, 4))
        grayscale_image.append(new_row)
        
    return grayscale_image
