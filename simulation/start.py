import numpy as np
import matplotlib.pyplot as plt

def simulate_debye_scherrer():
    # Experimental Parameters
    L = 1.5          # Distance from sample to film in cm
    wavelength_pm = 71.07 # Mo K-alpha (Leybold standard). Change to 154.06 for Cu.
    
    # Choline Chloride (Alpha form) Lattice Parameters (approximate) in pm
    a = 1121.0
    b = 1159.0
    c = 587.0
    
    # Main planes (h,k,l) and estimated relative intensities (approx)
    # Organic crystals often have strong low-angle peaks
    planes = [
        {'hkl': (1,1,0), 'I': 40},
        {'hkl': (0,2,0), 'I': 100}, # Often strong
        {'hkl': (2,0,0), 'I': 80},
        {'hkl': (1,2,0), 'I': 60},
        {'hkl': (0,1,1), 'I': 50},
        {'hkl': (1,1,1), 'I': 70},
        {'hkl': (0,0,2), 'I': 90},  # Strong peak at higher angle
        {'hkl': (1,2,1), 'I': 30},
        {'hkl': (2,1,1), 'I': 35},
        {'hkl': (0,2,2), 'I': 20},
    ]

    # Calculate Diffraction Angles and Radii
    r_values = []
    intensities = []
    labels = []

    print(f"{'Plane':<10} | {'2Theta (deg)':<12} | {'Radius r (cm)':<12}")
    print("-" * 40)

    for p in planes:
        h, k, l = p['hkl']
        # d-spacing formula for Orthorhombic
        inv_d2 = (h/a)**2 + (k/b)**2 + (l/c)**2
        d_pm = 1 / np.sqrt(inv_d2)
        
        # Bragg's Law: lambda = 2d sin(theta) -> sin(theta) = lambda / 2d
        sin_theta = wavelength_pm / (2 * d_pm)
        
        if sin_theta <= 1.0:
            theta_rad = np.arcsin(sin_theta)
            two_theta_rad = 2 * theta_rad
            two_theta_deg = np.degrees(two_theta_rad)
            
            # Radius on film: r = L * tan(2theta)
            r = L * np.tan(two_theta_rad)
            
            r_values.append(r)
            intensities.append(p['I'])
            labels.append(f"({h}{k}{l})")
            
            print(f"({h} {k} {l})    | {two_theta_deg:.2f}         | {r:.4f}")

    # Plotting
    x_sim = np.linspace(0, max(r_values)*1.2, 1000)
    y_sim = np.zeros_like(x_sim)

    # Add Gaussian peaks to simulate the graph
    sigma = 0.05 # Peak width parameter
    for r, I in zip(r_values, intensities):
        y_sim += I * np.exp(-0.5 * ((x_sim - r) / sigma)**2)

    plt.figure(figsize=(10, 6))
    plt.plot(x_sim, y_sim, color='blue', label='Simulated Choline Chloride (Mo Source)')
    
    # Add markers for specific planes
    for r, I, label in zip(r_values, intensities, labels):
        plt.text(r, I+2, label, ha='center', fontsize=8, rotation=90)

    plt.title(f"Simulated Debye-Scherrer Profile (L={L} cm, Mo Anode)")
    plt.xlabel("Distance from Center r (cm)")
    plt.ylabel("Relative Intensity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

simulate_debye_scherrer()