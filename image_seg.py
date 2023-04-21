import cv2
import numpy as np


img = cv2.imread('./images/v3s.png')
cv2.imshow('original', img)
cv2.waitKey()
cv2.destroyAllWindows()

# gausian filter
img = cv2.GaussianBlur(img, (25, 25), 0)
cv2.imshow('gausian', img)
cv2.waitKey()
cv2.destroyAllWindows()

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


lower_yellow = np.array([25, 80, 80])
upper_yellow = np.array([45, 255, 255])

lower_green = np.array([50, 100, 150])
upper_green = np.array([90, 255, 255])


mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
mask_green = cv2.inRange(hsv_img, lower_green, upper_green)

mask = cv2.bitwise_or(mask_yellow, mask_green)

result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('mask', result)
cv2.waitKey()
cv2.destroyAllWindows()


result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

cv2.imshow('gray mask', result)
cv2.waitKey()
cv2.destroyAllWindows()

# Apply Otsu's thresholding method
ret, thresh = cv2.threshold(result, 10, 255, cv2.THRESH_BINARY)

# Show the result
cv2.imshow('Thresholded Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((5, 5), np.uint8)
kernel2 = np.ones((29, 29), np.uint8)

# Perform opening operation
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# Perform closing operation
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)

# Show the result
cv2.imshow('Thresholded Image no noise', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Define the structuring element for dilation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
# Perform dilation to join small white areas
dilated = cv2.dilate(opening, kernel, iterations=1)
# Show the result
cv2.imshow('dilated', dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()


edges = cv2.Canny(dilated, 100, 200)
cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find contours in the edge image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Calculate the moments of each contour
for cnt in contours:
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    print('Center of contour:', (cx, cy))
    cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)

# Show the result
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
