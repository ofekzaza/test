import numpy as np
import itertools

# Your data as a NumPy array
data = np.array([
    [-2.35007496, -32.89036545, 3.58157113], [-11.91529235, -28.03156146, 3.06123817],
    [7.05022489, -27.3255814, 2.63155987], [-1.03073463, -19.10299003, 1.21503444],
    [11.04947526, -19.10299003, 1.61469299], [-14.76011994, -18.6461794, 1.87343792],
    [3.0922039, -17.6910299, 1.07128868], [-5.6071964, -17.52491694, 1.12433082],
    [8.90554723, -17.19269103, 1.24449542], [-11.7916042, -16.77740864, 1.39525544],
    [1.31934033, -14.95016611, 0.7489573], [-3.0922039, -14.78405316, 0.75851536],
    [6.06071964, -14.410299, 0.81242999], [-8.28710645, -14.03654485, 0.88306971],
    [17.02773613, -13.4551495, 1.56181856], [10.96701649, -12.7076412, 0.93627661],
    [-6.10194903, -11.91860465, 0.59643727], [-21.27436282, -11.5448505, 1.94038991],
    [-13.85307346, -11.46179402, 1.07375803], [15.25487256, -10.88039867, 1.16578387],
    [-0.57721139, -10.83887043, 0.39220154], [24.65517241, -10.67275748, 2.38695902],
    [9.48275862, -10.589701, 0.67204279], [-11.58545727, -9.88372093, 0.77105412],
    [2.72113943, -9.67607973, 0.33639319], [-3.71064468, -9.4269103, 0.34172914],
    [-18.8005997, -8.97009967, 1.43951016], [7.5862069, -8.59634551, 0.43752089],
    [13.1934033, -7.84883721, 0.78352077], [-8.86431784, -7.64119601, 0.45585401],
    [4.41154423, -7.35049834, 0.24477212], [-5.23613193, -6.97674419, 0.25342604],
    [-15.25487256, -5.89700997, 0.88898523], [11.58545727, -5.89700997, 0.56227133],
    [1.1131934, -5.81395349, 0.11675874], [-1.36056972, -5.77242525, 0.11719436],
    [16.49175412, -5.64784053, 1.00952307], [4.94752624, -5.27408638, 0.17421218],
    [6.84407796, -5.10797342, 0.24291263], [-5.40104948, -4.77574751, 0.17316371],
    [-7.5862069, -4.2358804, 0.25143333], [-12.6161919, -3.90365449, 0.58023382],
    [9.15292354, -3.73754153, 0.32546433], [14.10044978, -3.2807309, 0.69700023],
    [-18.63568216, -2.57475083, 1.17512359], [-9.93628186, -2.20099668, 0.34485054],
    [18.22338831, -1.57807309, 1.11115841], [5.44227886, -1.53654485, 0.10656005],
    [10.59595202, -0.87209302, 0.37631045], [-5.73088456, -0.49833887, 0.11026407],
    [-15.08995502, -0.45681063, 0.75780384], [5.73088456, 0.45681063, 0.11013195],
    [14.47151424, 0.66445183, 0.69793038], [-10.67841079, 1.28737542, 0.3851249],
    [-5.52473763, 2.03488372, 0.11550046], [-19.58395802, 2.15946844, 1.28844872],
    [9.73013493, 2.20099668, 0.33136703], [17.31634183, 2.57475083, 1.01816128],
    [7.66866567, 3.86212625, 0.2455472], [-14.67766117, 3.90365449, 0.76694683],
    [12.57496252, 3.94518272, 0.57786739], [5.68965517, 4.48504983, 0.17485758],
    [-8.988006, 4.56810631, 0.33845765], [2.18515742, 5.52325581, 0.11755816],
    [14.67766117, 5.56478405, 0.81909879], [-6.22563718, 5.85548173, 0.24328679],
    [-3.99925037, 5.89700997, 0.16913375], [0.08245877, 5.98006645, 0.1191793],
    [5.64842579, 6.47840532, 0.24604637], [-17.48125937, 6.81063123, 1.16871081],
    [-11.66791604, 7.01827243, 0.61672023], [9.07046477, 7.14285714, 0.44365636],
    [-3.42203898, 7.93189369, 0.24854505], [17.89355322, 8.22259136, 1.28711199],
    [4.53523238, 9.09468439, 0.34387788], [-13.31709145, 9.13621262, 0.86687942],
    [-6.84407796, 9.30232558, 0.44392532], [11.58545727, 9.38538206, 0.73920597],
    [-1.60794603, 9.92524917, 0.33660919], [19.70764618, 10.34053156, 1.6420717],
    [1.6904048, 10.67275748, 0.38871374], [13.60569715, 10.79734219, 1.00230989],
    [7.0089955, 11.37873754, 0.59416219], [-8.90554723, 11.87707641, 0.73278912],
    [-16.12068966, 12.83222591, 1.40852901], [-3.29835082, 13.20598007, 0.61632391],
    [8.90554723, 13.33056478, 0.85427647], [-28.03598201, 13.57973422, 3.20060531],
    [-10.22488756, 13.95348837, 0.9941991], [4.37031484, 14.28571429, 0.7421019],
    [-0.20614693, 14.90863787, 0.73921182], [-17.64617691, 15.44850498, 1.82240897],
    [-4.98875562, 15.49003322, 0.88018032], [12.28635682, 15.7807309, 1.32741335],
    [6.92653673, 17.10963455, 1.13145439], [15.00749625, 17.40033223, 1.74978255],
    [-1.85532234, 18.14784053, 1.10521611], [-7.83358321, 18.52159468, 1.34204471],
    [2.5149925, 19.10299003, 1.23243506], [-9.97751124, 20.9717608, 1.78723755],
    [12.73988006, 27.1179402, 2.96302568], [-5.44227886, 29.56810631, 2.98330402],
    [4.37031484, 33.55481728, 3.76939037]
])

import numpy as np
import itertools
import math

def get_lattice_params(basis_vectors):
    """Calculates reciprocal lattice parameters from three basis vectors."""
    if basis_vectors.shape != (3, 3):
        raise ValueError("Basis must be a 3x3 array.")
    
    v = basis_vectors
    a_star = np.linalg.norm(v[0])
    b_star = np.linalg.norm(v[1])
    c_star = np.linalg.norm(v[2])
    
    # Handle potential division by zero if a norm is 0
    if a_star == 0 or b_star == 0 or c_star == 0:
        return (0, 0, 0, 0, 0, 0)
    
    alpha_star = np.arccos(np.clip(np.dot(v[1], v[2]) / (b_star * c_star), -1.0, 1.0)) * 180 / np.pi
    beta_star = np.arccos(np.clip(np.dot(v[0], v[2]) / (a_star * c_star), -1.0, 1.0)) * 180 / np.pi
    gamma_star = np.arccos(np.clip(np.dot(v[0], v[1]) / (a_star * b_star), -1.0, 1.0)) * 180 / np.pi
    
    return (a_star, b_star, c_star, alpha_star, beta_star, gamma_star)

def check_system(params, ang_tol=2.0, len_tol=0.1):
    """
    Classifies lattice parameters into a crystal system.
    Tolerances are looser to allow for measurement error.
    """
    a, b, c, al, be, ga = params
    
    if a == 0: # Catch bad basis
        return "Triclinic" # Default to lowest symmetry

    def is_eq(v1, v2):
        return np.isclose(v1, v2, rtol=len_tol)
    
    def is_90(v):
        return np.isclose(v, 90, atol=ang_tol)

    def is_60(v):
        return np.isclose(v, 60, atol=ang_tol)

    # Check from most constrained to least constrained
    if is_eq(a, b) and is_eq(b, c) and is_90(al) and is_90(be) and is_90(ga):
        return "Cubic"
        
    if is_eq(a, b) and not is_eq(a, c) and is_90(al) and is_90(be) and is_90(ga):
        return "Tetragonal"
        
    # Reciprocal of Hexagonal has gamma* = 60
    if is_eq(a, b) and not is_eq(a, c) and is_90(al) and is_90(be) and is_60(ga):
         return "Hexagonal"
         
    if is_eq(a, b) and is_eq(b, c) and is_eq(al, be) and is_eq(be, ga) and not is_90(al):
        return "Trigonal"

    if not is_eq(a, b) and not is_eq(a, c) and not is_eq(b, c) and \
       is_90(al) and is_90(be) and is_90(ga):
        return "Orthorhombic"

    if (is_90(al) and is_90(ga) and not is_90(be)) or \
       (is_90(al) and is_90(be) and not is_90(ga)) or \
       (is_90(be) and is_90(ga) and not is_90(al)):
        return "Monoclinic"
        
    return "Triclinic" # Fits everything else

def score_basis(basis_vectors, all_data):
    """
    Calculates the RMS of the residuals (distance from nearest integer)
    for all hkl indices.
    """
    try:
        inv_basis = np.linalg.inv(basis_vectors)
    except np.linalg.LinAlgError:
        return np.inf, "Singular Basis", None # Return infinity for a bad basis
        
    hkl_matrix = all_data @ inv_basis
    residuals = hkl_matrix - np.round(hkl_matrix)
    rms_residual = np.sqrt(np.mean(residuals**2))
    
    params = get_lattice_params(basis_vectors)
    system = check_system(params)
    
    return rms_residual, system, params

def calculate_match_percent(rms, max_error=0.5):
    """
    Converts an RMS residual score to a match percentage.
    max_error = 0.5 (worst possible RMS)
    A perfect score (rms=0.0) is 100%.
    A worst-case score (rms=0.5) is 0%.
    """
    if rms == np.inf:
        return 0.0
    
    # Linearly scale the score
    match_percent = 100.0 * (1.0 - (rms / max_error))
    
    # Ensure percent is not negative
    return max(0.0, match_percent)


def find_best_fit(all_data):
    """
    Runs the main search loop to find the best-fitting basis for each system.
    """
    print(f"--- Starting Search ---")
    print(f"Testing all {math.comb(len(all_data), 3)} combinations...")
    
    # We want the HIGHEST percentage, so initialize to 0
    best_match_percent = {
        "Cubic": 0.0, "Tetragonal": 0.0, "Orthorhombic": 0.0,
        "Hexagonal": 0.0, "Trigonal": 0.0, "Monoclinic": 0.0, "Triclinic": 0.0
    }
    best_params_info = {}

    indices = range(len(all_data))
    
    for (i, j, k) in itertools.combinations(indices, 3):
        basis = np.array([all_data[i], all_data[j], all_data[k]])
        
        # Skip if basis is co-planar (determinant is near zero)
        if np.abs(np.linalg.det(basis)) < 1e-3:
            continue
            
        rms, system, params = score_basis(basis, all_data)
        
        if rms == np.inf:
            continue
            
        match_percent = calculate_match_percent(rms)
        
        # If this is the best (highest) % score for this system, save it
        if match_percent > best_match_percent[system]:
            best_match_percent[system] = match_percent
            best_params_info[system] = (params, rms)

    # --- Print Final Results ---
    print("\n--- Best Fit per System (Match Percentage) ---")
    print("A higher percentage is better.\n")

    print(f"{'System':<12} | {'Match %':<10} | {'(RMS Residual)':<15}")
    print("---------------------------------------------")
    
    # Sort by the best match percentage
    sorted_systems = sorted(best_match_percent, key=best_match_percent.get, reverse=True)
    
    for system in sorted_systems:
        score = best_match_percent[system]
        if score > 0:
            rms = best_params_info[system][1]
            print(f"{system:<12} | {score:<10.2f}% | (RMS: {rms:.4f})")
        else:
            print(f"{system:<12} | {score:<10.2f}% | {'(No solution found)'}")

    # --- Print Overall Best ---
    best_system_overall = sorted_systems[0]
    best_score_overall = best_match_percent[best_system_overall]
    
    if best_score_overall > 0:
        p, rms = best_params_info[best_system_overall]
        print("\n--- 🏆 Overall Best Fit ---")
        print(f"System: {best_system_overall}")
        print(f"Match %: {best_score_overall:.2f}%")
        print(f"RMS Residual: {rms:.4f} (A score near your 0.1 error is good)")
        print(f"Reciprocal Params (a*, b*, c*): {p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}")
        print(f"Reciprocal Angles (α*, β*, γ*): {p[3]:.1f}°, {p[4]:.1f}°, {p[5]:.1f}°")
    else:
        print("\nNo solution could be found in the search.")

# --- How to use this ---
# 1. Create your 'data' variable (the large NumPy array from your prompt)
# 2. Then, simply call the main function:
#
# data = np.array([...your 100 points...])
find_best_fit(data)
#