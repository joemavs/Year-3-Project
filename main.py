import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import exposure
from skimage.feature import canny
from skimage.io import imread
from skimage.color import rgb2gray
from math import atan2, degrees

# Load image and convert to grayscale
image_path = 'images/individual_tlc.jpg'
tlc_image = imread(image_path)

gray_image = rgb2gray(tlc_image)  # Converts to grayscale
rgb_array = np.array(tlc_image)

# Convert grayscale image to NumPy array
gray_array = np.array(gray_image)

# Generating figure 1
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes.ravel()

# Display grayscale image
ax[0].imshow(rgb_array)
ax[0].set_title('Image')
ax[0].set_axis_off()

# Apply bilateral filtering
smoothed_image_bilateral = cv2.bilateralFilter((gray_array * 255).astype(np.uint8), d=2, sigmaColor=100, sigmaSpace=75)

# Apply CLAHE
enhanced_image = exposure.equalize_adapthist(smoothed_image_bilateral, clip_limit=0.03)

# Apply Canny edge detection
edges = canny(enhanced_image, sigma=2)

# Convert Canny edges to uint8 format (required by OpenCV)
edges_uint8 = (edges * 255).astype(np.uint8)

# Apply Hough Line Transform
lines = cv2.HoughLinesP(edges_uint8, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

# Convert original image to RGB format (from grayscale)
image_with_lines = ((rgb_array * 255).astype(np.uint8))

# Define angle threshold for horizontal lines
angle_threshold = 10  # Acceptable deviation from 0° (horizontal)

# Draw only horizontal lines
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate the angle of the line
        angle = degrees(atan2(y2 - y1, x2 - x1))  # Compute angle in degrees

        # Keep only nearly horizontal lines
        if abs(angle) < angle_threshold or abs(angle - 180) < angle_threshold:
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Draw in blue

# Show detected horizontal lines
ax[1].imshow(image_with_lines)
ax[1].set_title('Detected Horizontal Lines')
ax[1].set_axis_off()

plt.show()

