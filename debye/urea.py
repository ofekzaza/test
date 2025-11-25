import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

# Load the user's image
image_path = './debye/IMG.jpg'
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Invert image so spots are bright (since they are dark in the original negative-style film)
# The original image has a dark center and dark spots on a lighter blue background.
# We want the dark spots to be high intensity peaks for detection.
inverted = 255 - gray

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to bring out the spots
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
enhanced = clahe.apply(inverted)

# Find the center of the diffraction pattern
# We assume the large dark blob (now bright) is the center.
# We smooth the image heavily to find the centroid of the main beam.
blurred = cv2.GaussianBlur(enhanced, (51, 51), 0)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred)
center_x, center_y = maxLoc

# Calculate Radial Profile (Azimuthal Integration)
y, x = np.indices(enhanced.shape)
r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
r = r.astype(np.int32)

# Bin the intensities by radius to find where the "rings" are strongest
tbin = np.bincount(r.ravel(), enhanced.ravel())
nr = np.bincount(r.ravel())
radial_profile = tbin / nr

# Identify peaks in the radial profile (ignoring the central beam)
# The central beam usually dominates the first few pixels.
start_radius = 50 # skip the central blob
end_radius = 400  # approximate extent of valid data
search_area = radial_profile[start_radius:end_radius]

# Simple peak detection: find radii with local maxima
from scipy.signal import find_peaks
peaks, _ = find_peaks(search_area, height=np.mean(search_area)*1.1, distance=20)
peak_radii = peaks + start_radius

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 1. Original Image with overlaid circles
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
for radius in peak_radii:
    circle = plt.Circle((center_x, center_y), radius, color='red', fill=False, linewidth=1.5, alpha=0.6)
    ax[0].add_artist(circle)
ax[0].scatter([center_x], [center_y], color='yellow', marker='x', s=100, label='Center')
ax[0].set_title("Urea Diffraction: Detected Rings Overlaid")
ax[0].axis('off')

# 2. Radial Intensity Profile
ax[1].plot(radial_profile)
ax[1].set_xlim(0, end_radius)
ax[1].set_title("Radial Intensity Profile")
ax[1].set_xlabel("Radius (pixels)")
ax[1].set_ylabel("Avg Intensity")
for radius in peak_radii:
    ax[1].axvline(x=radius, color='red', linestyle='--', alpha=0.5)
    ax[1].text(radius, ax[1].get_ylim()[1]*0.9, f'R={radius}', rotation=90, verticalalignment='top')

plt.tight_layout()
plt.savefig('urea_analysis.png')