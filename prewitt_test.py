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


# Load image and convert to grayscale
image_path = 'images/individual_tlc.jpg'
tlc_image = Image.open(image_path).convert('L')  # Change to scikit

# Convert grayscale image to NumPy array
image_array = np.array(tlc_image)

# Generating figure 1
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes.ravel()

# Apply bilateral filtering
smoothed_image_bilateral = cv2.bilateralFilter(image_array, d=2, sigmaColor=100, sigmaSpace=75)

# Apply CLAHE
enhanced_image = exposure.equalize_adapthist(smoothed_image_bilateral, clip_limit=0.03)

# Apply Canny edge detection
edges = canny(enhanced_image, sigma=2)

ax[0].imshow(edges, cmap='gray')
ax[0].set_title('Canny')
ax[0].set_axis_off()

tlc_prewitt_h = ndimage.prewitt(image_array, axis=0);
ax[1].imshow(tlc_prewitt_h, cmap='gray')
ax[1].set_title('Canny')
ax[1].set_axis_off()



plt.show()


