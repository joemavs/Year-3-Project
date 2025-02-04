from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from scipy.ndimage import median_filter
import cv2
from skimage import exposure
from skimage.feature import canny
from skimage.filters import threshold_local






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

ax[4].imshow(binary_image, cmap='gray')
ax[4].set_title('Canny')
ax[4].set_axis_off()



plt.show()


