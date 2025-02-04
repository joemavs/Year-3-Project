from PIL import Image
from skimage import color
from skimage.transform import hough_line,hough_line_peaks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.feature import canny




image_path = r'images/individual_tlc.jpg'
tlc_image = Image.open(image_path).convert('L')  # Convert to grayscale

# Convert to NumPy array
image_array = np.array(tlc_image)


# Classic straight-line Hough transform
# Set a precision of 0.5 degree.
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
# Apply Canny edge detection
edges = canny(image_array, sigma=0.001)  # Adjust sigma for edge detection sensitivity

h, theta, d = hough_line(edges, theta=tested_angles)

# Generating figure 1
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image_array, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(image_array, cmap=cm.gray)
ax[1].set_ylim((image_array.shape[0], 0))
ax[1].set_axis_off()
ax[1].set_title('Detected lines')

for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    ax[1].axline((x0, y0), slope=np.tan(angle + np.pi / 2))

plt.tight_layout()
plt.show()





