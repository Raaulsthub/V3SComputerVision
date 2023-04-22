import cv2
import numpy as np


def find_center(frame):
    original = frame
    # Apply gaussian filter
    img = cv2.GaussianBlur(frame, (25, 25), 0)

    # Convert to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create yellow and green color ranges and apply masks
    lower_yellow = np.array([25, 80, 80])
    upper_yellow = np.array([45, 255, 255])
    mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

    lower_green = np.array([50, 100, 150])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv_img, lower_green, upper_green)

    mask = cv2.bitwise_or(mask_yellow, mask_green)

    # Apply mask to original image
    result = cv2.bitwise_and(img, img, mask=mask)

    # Convert to grayscale and apply thresholding
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(result, 10, 255, cv2.THRESH_BINARY)

    # Apply opening and closing morphological operations
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, np.ones((29, 29), np.uint8))

    # Apply dilation to join small white areas
    dilated = cv2.dilate(opening, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=1)

    # Apply edge detection and find contours
    edges = cv2.Canny(dilated, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and calculate moments
    for cnt in contours:
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(original, (cx, cy), 10, (0, 0, 255), -1)

    return original