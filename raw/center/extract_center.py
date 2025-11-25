import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure
from skimage.measure import CircleModel, ransac
import os

def get_center_hybrid(image_path):
    img_rgb = io.imread(image_path)
    if img_rgb.shape[2] == 4:
        img_rgb = img_rgb[:, :, :3]
    gray = color.rgb2gray(img_rgb)
    
    filename = os.path.basename(image_path)
    
    # --- STRATEGY 1: Robust Shape Fit (For Liquid.jpg) ---
    if filename == 'Liquid.jpg':
        # Blur significantly to smooth the "fuzzy" gradient
        blurred = filters.gaussian(gray, sigma=5)
        # Use a standard threshold to catch the whole shadow shape, not just the core
        contours = measure.find_contours(blurred, 0.25)
        main_contour = max(contours, key=len)
        
        # RANSAC fits a perfect circle to the shape, ignoring irregularities
        model_robust, inliers = ransac(main_contour, CircleModel, min_samples=3,
                                       residual_threshold=2.0, max_trials=1000)
        cy, cx, radius = model_robust.params
        method = "Robust Shape Fit"

    # --- STRATEGY 2: Deepest Dark Core (For others) ---
    else:
        blurred = filters.gaussian(gray, sigma=2)
        min_val = np.min(blurred)
        # Strict threshold: only pixels near the absolute bottom darkness
        threshold_level = min_val + 0.05
        mask = blurred < threshold_level
        contours = measure.find_contours(mask, 0.5)
        
        if contours:
            core_contour = max(contours, key=lambda c: c.shape[0])
            model = CircleModel()
            model.estimate(core_contour)
            cy, cx, radius = model.params
            method = "Deepest Core"
        else:
            return img_rgb, 0, 0, 0, "Failed"

    return img_rgb, cx, cy, radius, method

# Run Process
image_files = ['Liquid.jpg', 'SALT.jpg', '10_1.jpg', 'eutentric.jpg', '7_20.jpg', 'Solid.jpg']

plt.figure(figsize=(12, 18))

for i, img_file in enumerate(image_files):
    img_file = f"raw/{img_file}"
    if os.path.exists(img_file):
        img, cx, cy, rad, method = get_center_hybrid(img_file)
        
        # Plot
        ax = plt.subplot(3, 2, i+1)
        
        # Zoom crop
        zoom = 150
        y1, y2 = max(0, int(cy)-zoom), min(img.shape[0], int(cy)+zoom)
        x1, x2 = max(0, int(cx)-zoom), min(img.shape[1], int(cx)+zoom)
        
        crop = img[y1:y2, x1:x2]
        cx_crop = cx - x1
        cy_crop = cy - y1
        
        ax.imshow(crop)
        
        # Yellow Circle
        circle = plt.Circle((cx_crop, cy_crop), rad, color='yellow', fill=False, linewidth=2)
        ax.add_patch(circle)
        
        # Red Crosshair
        ax.plot(cx_crop, cy_crop, 'r+', markersize=25, markeredgewidth=1.5)
        
        ax.set_title(f"{img_file}\nCenter: ({cx:.1f}, {cy:.1f})\nMethod: {method}")
        ax.axis('off')

plt.tight_layout()
plt.show()