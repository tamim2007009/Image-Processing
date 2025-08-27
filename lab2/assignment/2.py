import cv2

# Read the image in grayscale
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# Set the low and high thresholds for Canny
low_threshold = 50
high_threshold = 150

# Apply Canny edge detection
edges = cv2.Canny(img, low_threshold, high_threshold)

# Display the results
cv2.imshow("Original Image", img)
cv2.imshow("Canny Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
