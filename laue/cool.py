import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel

# --- Constants ---
# Physical dimensions of the image area in millimeters
PHYS_X_MM = 55.0
PHYS_Y_MM = 75.0
L = 150

import enum
class CrystalType(enum.Enum):
    FCC = "fcc"
    BCC = "bcc"
    SC = "simple"

class HKL(BaseModel):
    h: int
    k: int
    l: int
    crystal_type: CrystalType

def find_points(image_path: str):
    """
    Performs all the image processing logic to find point coordinates.
    
    Loads an image, finds the green center and all red dots,
    and calculates the relative pixel coordinates of red dots to the green center.
    
    Returns:
        tuple: (green_center, red_centers, relative_coords, image_shape, original_image)
               Returns (None, [], [], None, None) on failure.
    """
    # --- 1. Load Image ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, [], [], None, None

    image_shape = image.shape # (height, width, channels)
    print(f"Image loaded. Shape (H, W, C): {image_shape}")

    # --- 2. Convert to HSV Color Space ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- 3. Find Green Circle Center ---
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    green_center = None
    if contours_green:
        c_green = max(contours_green, key=cv2.contourArea)
        M = cv2.moments(c_green)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            green_center = (cX, cY)
    
    if green_center is None:
        print("Error: Could not find the green center circle.")
        return None, [], [], image_shape, image

    print(f"Green Center (Absolute): {green_center}")

    # --- 4. Find Red Points ---
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_centers = []
    for c_red in contours_red:
        if cv2.contourArea(c_red) > 5: # Filter out noise
            M = cv2.moments(c_red)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                red_centers.append((cX, cY))

    print(f"Found {len(red_centers)} red points.")

    # --- 5. Calculate Relative Pixel Coordinates ---
    relative_coords = []
    # Sort red_centers to make the table output consistent
    red_centers.sort(key=lambda p: (p[1], p[0]))
    
    for (rx, ry) in red_centers:
        relative_x = rx - green_center[0]
        relative_y = ry - green_center[1] 
        relative_coords.append((relative_x, relative_y))

    return green_center, red_centers, relative_coords, image_shape, image

def normalize_coordinates(relative_coords, image_shape, phys_x_mm, phys_y_mm) -> list[tuple[float, float, float]]:
    """
    Normalizes relative pixel coordinates to physical mm coordinates.
    """
    if image_shape is None:
        return []
        
    height, width = image_shape[:2]
    
    # Calculate mm-per-pixel scaling factors
    x_scale_mpp = phys_x_mm / width
    y_scale_mpp = phys_y_mm / height
    
    print(f"Image (H, W): ({height}, {width}) px")
    print(f"Physical (H, W): ({phys_y_mm}, {phys_x_mm}) mm")
    print(f"Scaling (Y, X): ({y_scale_mpp:.4f} mm/px, {x_scale_mpp:.4f} mm/px)")

    normalized_coords = []
    for (rel_x_px, rel_y_px) in relative_coords:
        # Apply scaling
        rel_x_mm = rel_x_px * x_scale_mpp
        rel_y_mm = rel_y_px * y_scale_mpp
        rel_z_mm = math.sqrt(rel_x_mm ** 2 + rel_y_mm ** 2 + L ** 2) - L 
        normalized_coords.append((rel_x_mm, rel_y_mm, rel_z_mm))
        
    return normalized_coords

def plot_points(image, green_center, red_centers, output_image_path='image_8bf385_detected.jpg', plot_path='detected_points_plot.png'):
    """
    Draws the detected points on a copy of the image and saves
    it, and also creates a matplotlib plot.
    """
    if image is None:
        print("Plotting skipped: No image provided.")
        return
    if green_center is None:
        print("Plotting skipped: No green center found.")
        return

    output_image = image.copy()

    # Draw Green Center
    (cX_g, cY_g) = green_center
    cv2.circle(output_image, (cX_g, cY_g), 10, (255, 255, 255), 2) # White circle
    cv2.line(output_image, (cX_g - 15, cY_g), (cX_g + 15, cY_g), (255, 255, 255), 2) # White H-line
    cv2.line(output_image, (cX_g, cY_g - 15), (cX_g, cY_g + 15), (255, 255, 255), 2) # White V-line

    # Draw Red Centers
    for (cX_r, cY_r) in red_centers:
        cv2.circle(output_image, (cX_r, cY_r), 5, (255, 255, 0), 2) # Cyan color
        
    # --- Save and Plot ---
    cv2.imwrite(output_image_path, output_image)
    print(f"\nSaved image with detected points to: {output_image_path}")

    # Display the image using matplotlib
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(output_image_rgb)
    plt.title("Detected Points (Green Center, Red Dots)")
    plt.axis('off') # Hide axes
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

def create_results_table_string(red_centers, relative_coords, normalized_coords, hkls: list[HKL]):
    """
    Combines all coordinate data into a formatted string table.
    """
    # Create header
    header = (
        f"{'Point #':<10} | "
        f"{'Abs X (px)':<12} | {'Abs Y (px)':<12} | "
        f"{'Rel X (px)':<12} | {'Rel Y (px)':<12} | "
        f"{'Rel X (mm)':<12} | {'Rel Y (mm)':<12} | {'Rel Z (mm)':<12}"
        f"{'h':<6} | {'k':<6} | {'l':<6} | {'type':<6}"
    )
    separator = "-" * len(header)
    
    lines = [header, separator]
    
    # Create rows
    for i, (rc, rel_c, norm_c, hkl) in enumerate(zip(red_centers, relative_coords, normalized_coords, hkls)):
        line = (
            f"{i + 1:<10} | "
            f"{rc[0]:<12} | {rc[1]:<12} | "
            f"{rel_c[0]:<12} | {rel_c[1]:<12} | "
            f"{norm_c[0]:<12.3f} | {norm_c[1]:<12.3f} |{norm_c[2]:<12.3f}"
            f"{hkl.h} | {hkl.k} | {hkl.l} | {hkl.crystal_type}"
        )
        lines.append(line)
        
    return "\n".join(lines)


def is_fcc(h, k, l):
    """
    Checks if Miller indices are 'unmixed' (all even or all odd).
    Zero is considered even.
    """
    h_even = h % 2 == 0
    k_even = k % 2 == 0
    l_even = l % 2 == 0

    all_even = h_even and k_even and l_even
    all_odd = (not h_even) and (not k_even) and (not l_even)

    return all_even or all_odd

def is_bcc(h,k,l) -> bool:
    return (h + k + l) % 2 == 0



def find_hkl(x, y, z) -> HKL:
    """
    Finds the smallest unmixed (h, k, l) triple based on the
    ratio x:y:z. Returns standard Python ints.
    """
    coords = np.array([x, y, z])

    non_zero_abs_coords = np.abs(coords[np.abs(coords) > 1e-6])

    if len(non_zero_abs_coords) == 0:
        return HKL(h=0, k=0, l=0, crystal_type=CrystalType.SC)

    min_val = np.min(non_zero_abs_coords)

    base_ratio = coords / min_val
    base_integers = np.round(base_ratio).astype(int)

    if np.all(base_integers == 0):
        base_integers = np.round(base_ratio * 10).astype(int)
        if np.all(base_integers == 0):
            return HKL(h=0, k=0, l=0, crystal_type=CrystalType.SC)

    h_base, k_base, l_base = base_integers

    for multiplier in range(1, 10):
        h = h_base * multiplier
        k = k_base * multiplier
        l = l_base * multiplier

        if is_fcc(h, k, l):
            return HKL(h=h, k=k, l=l, crystal_type=CrystalType.FCC)
        
        if is_bcc(h, k, l):
            return HKL(h=h, k=k, l=l, crystal_type=CrystalType.BCC)

    return HKL(h=h_base, k=k_base, l=l_base, crystal_type=CrystalType.SC)

# --- Main execution ---
if __name__ == "__main__":
    print("Attempting to process image...")

    # 1. Find points
    green_center, red_centers, relative_coords, image_shape, image = find_points(image_path='IMG.jpg')
    
    if green_center is None:
        print("Processing failed. Exiting.")
    else:
        # 2. Normalize coordinates
        normalized_coords = normalize_coordinates(relative_coords, image_shape, PHYS_X_MM, PHYS_Y_MM)
        
        hkls = [find_hkl(*cords) for cords in normalized_coords]

        # 3. Plot points
        plot_points(image, green_center, red_centers, output_image_path='image_8bf385_detected.jpg')
        
        # 4. Create and print table
        results_table_string = create_results_table_string(red_centers, relative_coords, normalized_coords, hkls)

        print("\n--- Coordinate Results Table ---")
        print(results_table_string)
        print(len([hkl for hkl in hkls if hkl.crystal_type == CrystalType.FCC]))
        print(len([hkl for hkl in hkls if hkl.crystal_type == CrystalType.BCC]))
        print(len(hkls))
        print(normalized_coords)

    print("Processing complete.")