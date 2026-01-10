import cv2
import numpy as np


def draw_bbox(
    image: np.ndarray,
    bbox: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    proportion: float = 0.2,
) -> None:
    """Draw a bounding box with fancy corners on an image.

    Args:
        image: Input image to draw on (modified in-place).
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        color: Color of the bounding box in BGR. Defaults to green.
        thickness: Thickness of the corner lines. Defaults to 3.
        proportion: Proportion of corner length to box dimensions. Defaults to 0.2.
    """
    x1, y1, x2, y2 = map(int, bbox)
    width = x2 - x1
    height = y2 - y1

    corner_length = int(proportion * min(width, height))

    # Draw the rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    # Top-left corner
    cv2.line(image, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + corner_length), color, thickness)

    # Top-right corner
    cv2.line(image, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + corner_length), color, thickness)

    # Bottom-left corner
    cv2.line(image, (x1, y2), (x1, y2 - corner_length), color, thickness)
    cv2.line(image, (x1, y2), (x1 + corner_length, y2), color, thickness)

    # Bottom-right corner
    cv2.line(image, (x2, y2), (x2, y2 - corner_length), color, thickness)
    cv2.line(image, (x2, y2), (x2 - corner_length, y2), color, thickness)
