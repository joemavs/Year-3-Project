import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imread
import cv2
from skimage import exposure
from skimage.feature import canny
from math import atan2, degrees

# Load image
image_path = 'images/individual_tlc.jpg'
tlc_image = imread(image_path)

# Convert image into numpy array
rgb_array = np.array(tlc_image)

def tellme(s):
    """
        Display a message as a title on the current figure and print it to the console.

        Parameters:
        s (str): The message to display.
        """
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()

def crop(rgb_array):
    """
        Allows the user to manually select a rectangular region in an image and crops it.

        The function displays an image, prompts the user to click on four corners of the
        desired region, and then crops the image accordingly.

        Parameters:
        rgb_array (numpy.ndarray): The input image array in RGB format.

        Returns:
        numpy.ndarray: An array of the cropped region of the image
        """
    plt.imshow(rgb_array)  # Shows image
    plt.title('Image')
    plt.axis('off')  # Hide axis for better visualization
    plt.draw()

    # Prompt the user to select four corners
    tellme("Please click on the corners of the TLC plate")
    corners = np.array(plt.ginput(4,0,True))   # Get four visible points from user input
    rounded_corners = corners.astype(int)  # Convert floating points to integers

    # Extract min and max x, y values from selected points to define the crop region
    x_min_crop = min([val[0] for val in rounded_corners])
    x_max_crop = max([val[0] for val in rounded_corners])
    y_min_crop = min([val[1] for val in rounded_corners])
    y_max_crop = max([val[1] for val in rounded_corners])

    # Perform cropping on array
    cropped_rgb_array = rgb_array[y_min_crop:y_max_crop, x_min_crop:x_max_crop]

    return cropped_rgb_array

# Get user to crop image
rgb_array = crop(rgb_array)

# Convert rgb array to grayscale array using Pillow
pil_image = Image.fromarray(rgb_array)  # Convert NumPy array to PIL image
gray_image = pil_image.convert('L')  # Convert to grayscale (L mode)
gray_array = np.array(gray_image)

# Apply bilateral filtering for noise reduction while preserving edges
smoothed_image_bilateral = cv2.bilateralFilter((gray_array * 255).astype(np.uint8), d=2, sigmaColor=100, sigmaSpace=75)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
enhanced_image = exposure.equalize_adapthist(smoothed_image_bilateral, clip_limit=0.03)

# Apply Canny edge detection to extract edges from the image
edges = canny(enhanced_image, sigma=2)

# Convert Canny edges to uint8 format (required by OpenCV)
edges_uint8 = (edges * 255).astype(np.uint8)

# Apply Hough Line Transform to detect lines in the edge-detected image
lines = np.squeeze(cv2.HoughLinesP(edges_uint8, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10))
image_with_lines = ((rgb_array * 255).astype(np.uint8))

# Define angle threshold to filter horizontal lines
angle_threshold = 10
horizontal_lines = []  # List to store detected horizontal lines


# Filter only horizontal lines
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line

        # Calculate the angle of the detected line
        angle = degrees(atan2(y2 - y1, x2 - x1))  # Compute angle in degrees

        # Keep only lines that are nearly horizontal
        if abs(angle) < angle_threshold or abs(angle - 180) < angle_threshold:
            horizontal_lines.append(line)

def grouplines(sortedlines):
    """
        Groups similar horizontal lines based on their y-coordinates.

        Lines that are within a small vertical range are grouped together.

        Parameters:
        sortedlines (list of arrays): List of detected horizontal lines.

        Returns:
        list: A list of grouped horizontal lines.
        """
    current_group = []
    groups = []  # List to store groups of similar lines
    line_indexes = list(range(len(sortedlines)))  # List of indices to track processed lines

    for i in range(len(line_indexes)):
        if line_indexes[i] == "null":
            # Skip lines that have already been processed
            continue
        current_group.append(sortedlines[i])  # Start a new group with the current line
        y_val = sortedlines[i][1]  # Get y-coordinate of the line

        # Compare with other lines to find similar ones
        for j in range(len(line_indexes)):
            if line_indexes[j] == "null" or i is j:
                # Skip lines already processed or the same line
                continue

            # If the y-coordinates are within ±7 pixels, consider them similar
            elif -7 < (sortedlines[j][1] - y_val) < 7:
                current_group.append(sortedlines[j])  # Add line to group
                line_indexes[j] = "null"  # Mark as processed

        line_indexes[i] = "null"  # Mark the current line as processed
        groups.append(current_group)  # Save grouped lines
        current_group = []
    return groups

def combinesimilarlines(groups,croplength):
    """
    Merges similar horizontal line groups into representative lines.

    If a group contains a single line, it is retained as is.
    If a group contains multiple lines, it averages their y-coordinates and 
    generates a new representative line spanning the full crop width.

    Parameters:
    groups (list of lists): A list of grouped horizontal lines.
                            Each group contains lines as numpy arrays [x1, y1, x2, y2].
    croplength (int): The width of the image crop (used to define the new line width).

    Returns:
    list: A list of simplified horizontal lines, each represented as [0, y, croplength, y].
    """
    lines = []  # Stores the merged/simplified lines

    for group in groups:
        if len(group) == 1:
            # If only one line in the group, use it directly (removing unnecessary dimensions)
            lines.append(np.squeeze(group))
        else:
            # Compute the average y-coordinate of the grouped lines
            y_vals = sum(line[1] for line in group)  # Sum all y-coordinates
            y_val = round(y_vals / len(group)) # Compute the average y-coordinate

            # Create a new horizontal line covering the full crop width
            lines.append(np.array([0,y_val,croplength,y_val], dtype=np.int32))

    return lines

def drawlines(lines,image):
    """
       Draws detected horizontal lines on an image.

       Parameters:
       lines (list of arrays): List of detected lines, each represented as [x1, y1, x2, y2].
       image (numpy.ndarray): The image on which lines will be drawn.
       """
    for line in lines:
        print("drawing line", line)
        x1, y1, x2, y2 = map(int, line)  # Ensure coordinates are integers
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)  # Draw in blue

    # Display image with lines
    plt.imshow(image)
    plt.show()

# Group horizontal lines
groups = grouplines(horizontal_lines)

# Merge similar lines
lines = combinesimilarlines(groups,100)

# Draw the final set of lines on the image
drawlines(lines,image_with_lines)