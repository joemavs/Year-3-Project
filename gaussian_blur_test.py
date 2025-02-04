from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from scipy.ndimage import median_filter
import cv2
from skimage import exposure
from skimage.feature import canny
from skimage.filters import threshold_local
from scipy import ndimage, datasets
from math import atan2, degrees




# Load image and convert to grayscale
image_path = 'images/individual_tlc.jpg'
tlc_image = Image.open(image_path).convert('L')  # Change to scikit

# Convert grayscale image to NumPy array
image_array = np.array(tlc_image)

# Generating figure 1
fig, axes = plt.subplots(1, 5, figsize=(15, 6))
ax = axes.ravel()

# Display grayscale image
ax[0].imshow(image_array, cmap='gray')
ax[0].set_title('Grayscale Image')
ax[0].set_axis_off()


# Apply bilateral filtering
smoothed_image_bilateral = cv2.bilateralFilter(image_array, d=2, sigmaColor=100, sigmaSpace=75)

ax[1].imshow(smoothed_image_bilateral, cmap='gray')
ax[1].set_title('Bilateral filter')
ax[1].set_axis_off()

# Apply CLAHE
enhanced_image = exposure.equalize_adapthist(smoothed_image_bilateral, clip_limit=0.03)
ax[2].imshow(enhanced_image, cmap='gray')
ax[2].set_title('CLAHE')
ax[2].set_axis_off()

# Apply Canny edge detection
edges = canny(enhanced_image, sigma=2)
ax[3].imshow(edges, cmap='gray')
ax[3].set_title('Canny')
ax[3].set_axis_off()

# Apply adaptive thresholding
block_size = 35  # Adjust based on image size
local_thresh = threshold_local(image_array, block_size, offset=10)
binary_image = enhanced_image > local_thresh

# Convert Canny edges to uint8 format (required by OpenCV)
edges_uint8 = (edges * 255).astype(np.uint8)

# Apply Hough Line Transform
lines = cv2.HoughLinesP(edges_uint8, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

# Convert original grayscale image to RGB for overlay
image_with_lines = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

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

ax[4].imshow(image_with_lines, cmap='gray')
ax[4].set_title('Canny')
ax[4].set_axis_off()


tlc_prewitt_h = ndimage.prewitt(image_array, axis=0);



plt.show()


