import numpy as np
import itertools
import math

# --- 1. CORE LOGIC: THE SCORING FUNCTION (NOW CORRECTED) ---

def score_basis(basis_vectors, all_data):
    """
    Calculates the RMS residual for a basis against all data points.
    
    *** This is the new, "smart" scorer. ***
    It checks for two possibilities:
    1. A Primitive (P) lattice: h,k,l are all integers.
    2. A Centered (I/F) lattice: h,k,l are all integers OR all half-integers.
    
    It returns the *minimum* (best) of these two scores.
    """
    try:
        inv_basis = np.linalg.inv(basis_vectors)
    except np.linalg.LinAlgError:
        return np.inf, "Singular Basis", None
        
    # Calculate h,k,l for all 100 original points
    hkl_matrix = all_data @ inv_basis
    
    # --- THIS IS THE CRITICAL NEW LOGIC ---
    
    # 1. Test for Primitive (P) lattice (h,k,l are all integers)
    # Residuals are (e.g.) 1.1 -> 0.1, 1.9 -> -0.1
    residuals_P = hkl_matrix - np.round(hkl_matrix)
    rms_P = np.sqrt(np.mean(residuals_P**2))
    
    # 2. Test for Centered (I/F) lattice (h,k,l are all integers OR all half-integers)
    # We multiply by 2. 
    # (e.g., 0.5 -> 1.0, 1.0 -> 2.0, 1.5 -> 3.0)
    # This maps all integers AND half-integers to new integers.
    hkl_matrix_centered = hkl_matrix * 2
    
    # Residuals are now (e.g.) 3.1 -> 0.1, 2.9 -> -0.1
    residuals_C = hkl_matrix_centered - np.round(hkl_matrix_centered)
    
    # We must divide the final RMS by 2 to get it back to the original scale
    rms_C = np.sqrt(np.mean(residuals_C**2)) / 2.0
    
    # The true residual is the *minimum* of these two possibilities
    # For your NaCl data, rms_C will be low (good!) and rms_P will be high (bad).
    rms_residual = min(rms_P, rms_C)
    
    # --- END CRITICAL LOGIC ---
    
    params = get_lattice_params(basis_vectors)
    system = check_system(params)
    
    return rms_residual, system, params


# --- 2. HELPER FUNCTIONS (CLASSIFICATION & MATH) ---

def get_lattice_params(basis_vectors):
    """Calculates reciprocal lattice parameters from three basis vectors."""
    if basis_vectors.shape != (3, 3):
        raise ValueError("Basis must be a 3x3 array.")
    
    v = basis_vectors
    a_star = np.linalg.norm(v[0])
    b_star = np.linalg.norm(v[1])
    c_star = np.linalg.norm(v[2])
    
    # Handle potential division by zero if a norm is 0
    if a_star < 1e-9 or b_star < 1e-9 or c_star < 1e-9:
        return (0, 0, 0, 0, 0, 0)
    
    alpha_star = np.arccos(np.clip(np.dot(v[1], v[2]) / (b_star * c_star), -1.0, 1.0)) * 180 / np.pi
    beta_star = np.arccos(np.clip(np.dot(v[0], v[2]) / (a_star * c_star), -1.0, 1.0)) * 180 / np.pi
    gamma_star = np.arccos(np.clip(np.dot(v[0], v[1]) / (a_star * b_star), -1.0, 1.0)) * 180 / np.pi
    
    return (a_star, b_star, c_star, alpha_star, beta_star, gamma_star)

def check_system(params, ang_tol=5.0, len_tol=1):
    """Classifies lattice parameters into a crystal system."""
    a, b, c, al, be, ga = params
    
    if a == 0: return "Triclinic" # Catch bad basis

    def is_eq(v1, v2): return np.isclose(v1, v2, rtol=len_tol)
    def is_90(v): return np.isclose(v, 90, atol=ang_tol)
    def is_60(v): return np.isclose(v, 60, atol=ang_tol)

    if is_eq(a, b) and is_eq(b, c) and is_90(al) and is_90(be) and is_90(ga):
        return "Cubic"
    if is_eq(a, b) and not is_eq(a, c) and is_90(al) and is_90(be) and is_90(ga):
        return "Tetragonal"
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
    return "Triclinic"

def calculate_match_percent(rms, max_error=0.5):
    """Converts an RMS residual score (0.0=perfect, 0.5=worst) to a match percentage."""
    if rms == np.inf: return 0.0
    match_percent = 100.0 * (1.0 - (rms / max_error))
    return max(0.0, match_percent)

# --- 3. THE MAIN SEARCH FUNCTION (DVM) ---

def find_best_fit_dvm(original_data, tolerance=0.1, search_n=30):
    """
    Finds the best crystal system using the Difference Vector Method (DVM).
    
    :param original_data: The (N, 3) array of your 100 points.
    :param tolerance: Your measurement error (0.1). Used to "uniquify" vectors.
    :param search_n: How many short vectors to use in the search. 
                       (30-50 is a good range).
    """
    
    # --- Step 1: Generate All Difference Vectors ---
    print(f"Generating difference vectors from {len(original_data)} points...")
    diff_vectors = []
    for p1, p2 in itertools.combinations(original_data, 2):
        diff_vectors.append(p1 - p2)
        diff_vectors.append(p2 - p1)
    
    print(f"Generated {len(diff_vectors)} difference vectors.")
    
    # --- Step 2: Find the Shortest *Unique* Vectors ---
    # This is the key. We find the shortest vectors in the new list,
    # and we "uniquify" them using your 0.1 tolerance.
    
    # Sort all 9,900 vectors by length (shortest first)
    diff_vectors.sort(key=np.linalg.norm)
    
    unique_short_vectors = []
    for v in diff_vectors:
        # Skip zero-length vectors
        if np.linalg.norm(v) < 1e-6:
            continue
            
        # Check if this vector is just a "duplicate" of one we already have
        is_duplicate = False
        for u in unique_short_vectors:
            # Check if v is "too close" to u
            if np.linalg.norm(v - u) < tolerance:
                is_duplicate = True
                break
        
        # If it's not a duplicate, add it to our list
        if not is_duplicate:
            unique_short_vectors.append(v)
            
        # Stop once we have enough vectors for our search
        if len(unique_short_vectors) >= search_n:
            break
            
    print(f"Found {len(unique_short_vectors)} unique short vectors for the search.")
    
    if len(unique_short_vectors) < 3:
        print("Error: Could not find 3 non-coplanar vectors. Data may be 1D or 2D.")
        return

    # --- Step 3: Search Combinations of *These* Vectors ---
    
    # We want the HIGHEST percentage, so initialize to 0
    best_match_percent = {
        "Cubic": 0.0, "Tetragonal": 0.0, "Orthorhombic": 0.0,
        "Hexagonal": 0.0, "Trigonal": 0.0, "Monoclinic": 0.0, "Triclinic": 0.0
    }
    best_params_info = {}

    num_combinations = math.comb(len(unique_short_vectors), 3)
    print(f"Testing {num_combinations} combinations (this is the real search)...")

    for (v1, v2, v3) in itertools.combinations(unique_short_vectors, 3):
        basis = np.array([v1, v2, v3])
        
        # Skip if basis is co-planar (determinant is near zero)
        if np.abs(np.linalg.det(basis)) < 1e-3:
            continue
            
        # --- Step 4: Score and Classify ---
        # This now calls our new, "smart" scorer
        rms, system, params = score_basis(basis, original_data)
        
        if rms == np.inf:
            continue
            
        match_percent = calculate_match_percent(rms)
        
        if match_percent > best_match_percent[system]:
            best_match_percent[system] = match_percent
            best_params_info[system] = (params, rms)

    # --- Step 5: Print Final Results ---
    print("\n--- 📊 Best Fit per System (Match Percentage) ---")
    print("A higher percentage is better.\n")

    print(f"{'System':<12} | {'Match %':<10} | {'(RMS Residual)':<15}")
    print("---------------------------------------------")
    
    sorted_systems = sorted(best_match_percent, key=best_match_percent.get, reverse=True)
    
    for system in sorted_systems:
        score = best_match_percent[system]
        if score > 0.01: # Only show non-trivial results
            rms = best_params_info[system][1]
            print(f"{system:<12} | {score:<10.2f}% | (RMS: {rms:.4f})")
        else:
            print(f"{system:<12} | {score:<10.2f}% | {'(No good fit found)'}")

    best_system_overall = sorted_systems[0]
    best_score_overall = best_match_percent[best_system_overall]
    
    if best_score_overall > 0.01:
        p, rms = best_params_info[best_system_overall]
        print("\n--- 🏆 Overall Best Fit ---")
        print(f"System: {best_system_overall}")
        print(f"Match %: {best_score_overall:.2f}%")
        print(f"RMS Residual: {rms:.4f} (A score near your {tolerance} error is good)")
        print(f"Reciprocal Params (a*, b*, c*): {p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}")
        print(f"Reciprocal Angles (α*, β*, γ*): {p[3]:.1f}°, {p[4]:.1f}°, {p[5]:.1f}°")
    else:
        print("\nNo solution could be found in the search.")

from fit_system import data


# 3. Run the analysis.
# We will use your 0.1 error as the tolerance.
# We will search the 30 shortest unique vectors.
find_best_fit_dvm(data, tolerance=0.1, search_n=30)