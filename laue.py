
#!/usr/bin/env python

# --- NEW LINES TO FIX LAUETOOLS ---
import os
import LaueTools
# Get the root directory of the installed LaueTools package
laue_tools_path = LaueTools.__path__[0]
# Set the environment variable that LaueTools needs
os.environ['LAUEPYTHONPATH'] = laue_tools_path
# ------------------------------------
#!/usr/bin/env python

"""
===========================================================================
 Laue Pattern Bravais Lattice Determination Script (Corrected)
===========================================================================

 This script performs a full analysis of raw (x,y) Laue spot data to
 determine the statistically most probable Bravais lattice.

 The workflow is as follows:
 1.  Load raw (x,y) data and CRITICAL experimental parameters.
 2.  Convert (x,y) pixel coordinates to physical (2theta, chi) angles,
     creating a 'spots.cor' file (requires 'lauetools').
 3.  Index the spot pattern against prototype lattices (e.g., 'Si') to
     find candidate orientation matrices (requires 'lauetools').
 4.  Refine the 6 unit cell parameters (a,b,c,alpha,beta,gamma) for
     each candidate solution to get the best-fit metric cell
     (requires 'lauetools').
 5.  Filter solutions based on statistical quality (Metric 1: % indexed
     spots and mean angular error).
 6.  Perform a rigorous lattice symmetry analysis on the single best
     refined unit cell to find the most probable Bravais lattice
     (requires 'cctbx' and 'pymatgen').
 7.  Print a final statistical report.

 CRITICAL: This script CANNOT work without accurate experimental
           parameters entered in the 'CRITICAL_PARAMETERS' dictionary.
           The 'dd' value has been set to 15.0 mm, but all other
           values (xcen, ycen, etc.) are placeholders and MUST be edited.
"""

import numpy as np
import os
import warnings

# --- Library Imports ---
# These libraries must be installed:
# pip install numpy matplotlib lauetools cctbx-base pymatgen

try:
    import LaueTools.LaueGeometry as LG
    import LaueTools.IOimagefile as IOimage
    import LaueTools.CrystalParameters as CP
    import LaueTools.IndexSpotsSet as ISS
    import LaueTools.generaltools as GT
    import LaueTools.fitOrient as FO
except ImportError:
    print("Error: 'lauetools' library not found.")
    print("Please install it: pip install lauetools")
    exit()

try:
    from cctbx import crystal
    from cctbx.sgtbx.lattice_symmetry import metric_subgroups
except ImportError:
    print("Error: 'cctbx-base' library not found.")
    print("Please install it: pip install cctbx-base")
    exit()

try:
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:
    print("Error: 'pymatgen' library not found.")
    print("Please install it: pip install pymatgen")
    exit()

# Suppress runtime warnings from libraries
warnings.filterwarnings("ignore")


# =========================================================================
# --- 1. USER-DEFINED PARAMETERS ---
# =========================================================================

# --- CRITICAL EXPERIMENTAL GEOMETRY ---
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!! YOU MUST EDIT THE PLACEHOLDER VALUES. 'dd' has been set to 15.0 mm.!!!
#!!! The script will fail or give incorrect results if 'xcen', 'ycen',!!!
#!!! 'pixelsize', etc., are wrong.                                      !!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CRITICAL_PARAMETERS = {
    'dd': 15.0,           # Sample-to-detector distance (mm) [1]
    'xcen': 941.76,       # Beam center X (pixels) [1]
    'ycen': 1082.44,      # Beam center Y (pixels) [1]
    'xbet': 0.629,        # Detector tilt angle (deg) [1]
    'xgam': -0.685,       # Detector tilt angle (deg) [1]
    'pixelsize': 0.07914, # Pixel size (mm) [1, 2]
    'kf_direction': 'Z>0',# Geometry (e.g., 'Z>0' for back-reflection) [2]
    'framedim': (2048, 2048) # Detector dimensions (pixels) [2]
}

# --- PROTOTYPE LATTICES FOR INDEXING SEARCH ---
# The script will try to index using these known materials.
# Add more if needed (e.g., 'W', 'GaAs', 'Graphite', 'Cu')
# 'Si' is a good default for many cubic/diamond structures.
# **FIX 1: Initialized empty list**
PROTOTYPE_MATERIALS = ['Si'] # Added 'Si' as a default guess

# --- ENERGY RANGE ---
# The min and max energy (in keV) of your polychromatic beam. [3]
ENERGY_RANGE_KEV = (5.0, 22.0) # This is (E_min, E_max)

# --- STATISTICAL THRESHOLDS ---
# These values define a "good" solution.
STATISTICAL_THRESHOLDS = {
    'min_spots_indexed_pct': 0.70,   # Must index at least 70% of spots
    'max_mean_error_deg': 0.5,     # Mean angular error must be < 0.5 deg
    'cctbx_tolerance_deg': 2.0     # Tolerance for lattice symmetry search
}

# --- USER-PROVIDED SPOT DATA ---
USER_SPOTS_XY = np.array([
    (4.255, -23.632), (-1.445, -19.03), (10.237, -18.947), (-14.533, -18.74),
    (2.891, -17.62), (-5.661, -17.496), (8.31, -17.081), (0.923, -14.884),
    (-3.172, -14.677), (5.741, -14.469), (-5.259, -14.096), (-4.416, -14.096),
    (-8.23, -14.013), (-6.102, -13.433), (16.179, -13.267), (-7.708, -12.811),
    (10.438, -12.562), (-20.836, -11.484), (-13.449, -11.484), (14.814, -10.945),
    (-0.723, -10.779), (2.529, -9.701), (-3.854, -9.411), (-16.46, -8.872),
    (7.266, -8.582), (-8.792, -7.587), (0.361, -7.463), (4.175, -7.338),
    (-5.219, -6.924), (-14.894, -5.846), (-13.449, -5.804), (15.737, -5.597),
    (14.131, -5.556), (14.412, -3.939), (8.712, -3.773), (-14.774, -3.027),
    (14.332, -2.653), (-18.226, -2.612), (14.774, -2.57), (-9.715, -2.197),
    (15.055, -2.114), (14.894, -1.824), (17.423, -1.658), (10.077, -0.871),
    (-8.19, -0.207), (-10.558, 1.285), (-19.19, 2.197), (9.234, 2.197),
    (16.661, 2.612), (-14.734, 2.902), (14.573, 3.607), (7.266, 3.897),
    (13.81, 4.27), (-8.912, 4.519), (14.091, 5.431), (-6.223, 5.763),
    (-13.73, 5.929), (5.38, 6.468), (-13.891, 6.551), (-17.102, 6.758),
    (8.672, 7.131), (-3.533, 7.877), (4.255, 8.955), (-13.088, 8.997),
    (11.321, 9.287), (-6.825, 9.328), (-1.726, 9.95), (19.029, 10.282),
    (12.847, 10.614), (1.445, 10.655), (-8.953, 11.94), (8.511, 13.267),
    (-10.478, 14.055), (3.854, 14.345), (-6.624, 14.884), (-0.201, 14.967),
    (-5.058, 15.257), (-17.383, 15.423), (6.343, 17.04), (14.292, 17.454),
    (-2.088, 18.242), (2.369, 19.03), (-10.036, 20.937)
])


# =========================================================================
# --- 2. ANALYSIS FUNCTIONS (Code Blocks from Report) ---
# =========================================================================

# --- Block 2 Function ---
def step1_convert_xy_to_cor(xy_data, calibration_dict, dat_file="spots.dat", cor_file="spots.cor"):
    print(f"Step 1: Converting (x,y) pixels to (2theta, chi) angles...")
    print("  WARNING: No intensity data provided. Assuming uniform intensity (100.0) for all spots.")
    
    intensities = np.full((len(xy_data), 1), 100.0)
    dat_array = np.hstack((xy_data, intensities))
    
    header = "X Y I"
    np.savetxt(dat_file, dat_array, fmt='%.3f', header=header, comments='')
    
    try:
        # LaueTools function to convert.dat to.cor [2]
        LG.convert2corfile(dat_file,
                           calibration_dict,
                           CCDCalibdict=calibration_dict)
        
        print(f"  Successfully converted {dat_file} to {cor_file}")
        return cor_file
    except Exception as e:
        print(f"\nFATAL ERROR in Step 1: Failed to convert.dat to.cor.")
        print(f"  This is likely due to an ERROR in the 'CRITICAL_PARAMETERS' dictionary.")
        print(f"  Please verify all detector parameters, especially 'xcen' and 'ycen'.")
        print(f"  Error details: {e}")
        return None

# --- Block 3 Function ---
def step2_index_prototypes(cor_file, prototype_materials, energy_range_keV):
    print(f"\nStep 2: Indexing pattern against {len(prototype_materials)} prototype(s)...")
    # **FIX 2: Initialized empty list**
    all_candidate_matrices = []
    
    spots_data = GT.Read_data(cor_file)
    if spots_data is None or len(spots_data) == 0:
        print(f"  ERROR in Step 2:.cor file '{cor_file}' is empty or could not be read.")
        return [] # Return empty list

    num_spots = len(spots_data)
    print(f"  Loaded {num_spots} spots from {cor_file}")
    
    # Unpack energy range
    E_min, E_max = energy_range_keV

    for mat_name in prototype_materials:
        print(f"  Trying prototype: {mat_name}")
        try:
            crystal = CP.GetCrystalParams(mat_name)
            if crystal is None:
                print(f"    Warning: Prototype '{mat_name}' not in LaueTools database. Skipping.")
                continue
                
            # **FIX 3: Unpacked energy_range_keV into E_min, E_max**
            iss = ISS.IndexSpotsSet(spots_data,
                                    crystal,
                                    E_min,
                                    E_max)

            # Set indexing parameters [4]
            iss.ISSS = min(30, num_spots)
            iss.angtol = 0.5
            
            iss.FindOrientMatrices()

            for matrix in iss.Final_UBmat:
                all_candidate_matrices.append({
                    'matrix': matrix,
                    'prototype': mat_name,
                    'crystal': crystal
                })
            print(f"    Found {len(iss.Final_UBmat)} candidate solution(s) for {mat_name}.")
            
        except Exception as e:
            print(f"    Error while indexing with {mat_name}: {e}")
            
    print(f"  Found {len(all_candidate_matrices)} total candidate solutions.")
    return all_candidate_matrices

# --- Block 4 Function ---
def step3_refine_cells(candidate_solutions, cor_file, energy_range_keV, detector_params):
    print(f"\nStep 3: Refining unit cell parameters for {len(candidate_solutions)} candidate(s)...")
    # **FIX 4: Initialized empty list**
    refined_solutions = []
    
    spots = GT.Read_data(cor_file)
    if spots is None:
        return [] # Return empty list
    
    # Unpack energy range
    E_min, E_max = energy_range_keV

    for i, candidate in enumerate(candidate_solutions):
        print(f"  Refining solution {i+1}/{len(candidate_solutions)} (from {candidate['prototype']})...")
        try:
            # Refine cell parameters, orientation, and detector parameters [5]
            # **FIX 5: Unpacked energy_range_keV into E_min, E_max**
            refined_dict, indexed_spots, _ = FO.fit_cell_and_orientation(
                candidate['matrix'],
                spots,
                candidate['crystal'],
                E_min,
                E_max,
                detector_params=detector_params,
                fitcell=True,
                fitU=True,
                verbose=False)
            
            indexed_count = len(indexed_spots)
            total_spots = len(spots)
            mean_angular_error = refined_dict['ang_residue']
            # new_cell is [a, b, c, alpha, beta, gamma]
            refined_cell_params = refined_dict['new_cell']
            indexed_hkls = refined_dict['hkl_indexed']

            refined_solutions.append({
                'cell_params': refined_cell_params,
                'indexed_count': indexed_count,
                'total_spots': total_spots,
                'mean_error_deg': mean_angular_error,
                'prototype': candidate['prototype'],
                'indexed_hkls': indexed_hkls
            })
            
            # **FIX 6: Correctly indexed 6-element list (0 to 5)**
            print(f"    Refined Cell: a={refined_cell_params[0]:.3f}, b={refined_cell_params[1]:.3f}, c={refined_cell_params[2]:.3f}, "
                  f"alpha={refined_cell_params[3]:.2f}, beta={refined_cell_params[4]:.2f}, gamma={refined_cell_params[5]:.2f}")
            print(f"    Fit Quality: {indexed_count}/{total_spots} spots indexed. Mean Error: {mean_angular_error:.4f} deg.")
            
        except Exception as e:
            print(f"    Refinement failed for solution {i+1}: {e}")

    # Sort solutions by best quality (most spots, then lowest error)
    sorted_solutions = sorted(refined_solutions,
                              key=lambda x: (-x['indexed_count'], x['mean_error_deg']))
    
    print(f"  Refinement complete. Found {len(sorted_solutions)} viable solutions.")
    return sorted_solutions

# --- Block 5 Function ---
def step4_get_bravais_stats(refined_cell_params, tolerance_deg=2.0):
    print(f"\nStep 4: Performing statistical lattice symmetry analysis...")
    cp = refined_cell_params
    # **FIX 7: Correctly indexed 6-element list (0 to 5)**
    print(f"  Input Cell: a={cp[0]:.4f}, b={cp[1]:.4f}, c={cp[2]:.4f}, "
          f"alpha={cp[3]:.3f}, beta={cp[4]:.3f}, gamma={cp[5]:.3f}")

    analysis_results = {}

    # 1. Pymatgen high-level analysis [11, 12]
    try:
        lattice = Lattice.from_parameters(*refined_cell_params)
        # **FIX 8: Added dummy atom coordinate [[0,0,0]] for Pymatgen**
        structure = Structure(lattice, ["H"], [[0, 0, 0]])
        sga = SpacegroupAnalyzer(structure, symprec=0.1) # symprec is distance tolerance
        best_system = sga.get_crystal_system()
        best_lattice_type = sga.get_lattice_type()
        analysis_results['pymatgen_system'] = best_system
        analysis_results['pymatgen_lattice'] = best_lattice_type
    except Exception as e:
        print(f"  Pymatgen analysis failed: {e}")
        analysis_results['pymatgen_system'] = "Unknown"
        analysis_results['pymatgen_lattice'] = "Unknown"

    # 2. CCTBX detailed statistical fit analysis [13]
    try:
        input_symmetry = crystal.symmetry(
            unit_cell=refined_cell_params,
            space_group_symbol="P1"
        )
        
        # This is the core statistical function [13]
        subgroups = metric_subgroups(
            input_symmetry=input_symmetry,
            max_delta=tolerance_deg
        )

        # **FIX 9: Initialized empty list**
        cctbx_fits = []
        for group_info in subgroups.result_groups:
            cctbx_fits.append({
                'bravais_type': group_info['bravais_type'],
                'fit_cell': group_info['best_cell'].parameters(),
                'score_max_delta_deg': group_info['max_angular_difference']
            })
        
        cctbx_fits_sorted = sorted(cctbx_fits, key=lambda x: x['score_max_delta_deg'])
        analysis_results['cctbx_fits'] = cctbx_fits_sorted
        
    except Exception as e:
        print(f"  CCTBX analysis failed: {e}")
        # **FIX 10: Initialized empty list on failure**
        analysis_results['cctbx_fits'] = []

    return analysis_results


# =========================================================================
# --- 3. MAIN WORKFLOW ---
# =========================================================================
def main():
    print("======================================================")
    print(" Starting Laue Bravais Lattice Analysis")
    print("======================================================")

    # --- Step 1 ---
    cor_file = step1_convert_xy_to_cor(USER_SPOTS_XY, CRITICAL_PARAMETERS)
    if cor_file is None:
        return

    # --- Step 2 ---
    candidates = step2_index_prototypes(cor_file, PROTOTYPE_MATERIALS, ENERGY_RANGE_KEV)
    if not candidates:
        print("\nANALYSIS FAILED: No candidate orientation matrices were found.")
        print("Possible Reasons:")
        print(" 1. The 'CRITICAL_PARAMETERS' are incorrect (most likely).")
        print(" 2. The 'PROTOTYPE_MATERIALS' list does not include a good starting guess.")
        print(" 3. The data is not from a single crystal.")
        return

    # --- Step 3 ---
    # We need the detector parameters in a list format for fitOrient
    # **FIX 11: Correctly created the 5-element list**
    det_params = [CRITICAL_PARAMETERS['dd'], CRITICAL_PARAMETERS['xcen'],
                  CRITICAL_PARAMETERS['ycen'], CRITICAL_PARAMETERS['xbet'],
                  CRITICAL_PARAMETERS['xgam']]
    
    solutions = step3_refine_cells(candidates, cor_file, ENERGY_RANGE_KEV, det_params)
    if not solutions:
        print("\nANALYSIS FAILED: Refinement failed for all candidate solutions.")
        return

    # --- Step 4: Filter and Analyze Best Solution ---
    # **FIX 12: Selected the BEST solution (index 0) from the sorted list**
    best_solution = solutions[0]
    indexed_pct = best_solution['indexed_count'] / best_solution['total_spots']
    
    print("\n------------------------------------------------------")
    print("---           FINAL ANALYSIS REPORT                ---")
    print("------------------------------------------------------")

    # --- Metric 1: Indexing Goodness-of-Fit ---
    print("\n--- Metric 1: Indexing Goodness-of-Fit ---")
    print(f"  Best Solution Source:     '{best_solution['prototype']}' prototype")
    print(f"  Indexed Spots:          {best_solution['indexed_count']} / {best_solution['total_spots']} ({indexed_pct:.1%})")
    print(f"  Mean Angular Error:     {best_solution['mean_error_deg']:.4f} degrees")

    # Apply statistical filters
    if (indexed_pct < STATISTICAL_THRESHOLDS['min_spots_indexed_pct'] or
        best_solution['mean_error_deg'] > STATISTICAL_THRESHOLDS['max_mean_error_deg']):
        print("\nWARNING: Best solution is POOR QUALITY.")
        print(f"  It failed to meet the thresholds (Min {STATISTICAL_THRESHOLDS['min_spots_indexed_pct']:.0%} spots, Max {STATISTICAL_THRESHOLDS['max_mean_error_deg']:.2f} deg error).")
        print("  The resulting lattice analysis may be incorrect.")

    # --- Metric 2: Bravais Lattice Statistical Fit ---
    print("\n--- Metric 2: Bravais Lattice Statistical Fit ---")
    analysis = step4_get_bravais_stats(
        best_solution['cell_params'],
        STATISTICAL_THRESHOLDS['cctbx_tolerance_deg']
    )
    
    # --- Final Conclusion ---
    print("\n--- CONCLUSION ---")
    if not analysis.get('cctbx_fits'):
        print("Lattice symmetry analysis failed. Cannot determine Bravais lattice.")
        return

    pymatgen_guess = f"{analysis['pymatgen_system']} ({analysis['pymatgen_lattice']})"
    # **FIX 13: Selected the BEST fit (index 0) from the sorted list**
    best_cctbx_fit = analysis['cctbx_fits'][0]
    
    print(f"  Pymatgen High-Level ID: {pymatgen_guess}")
    print(f"  CCTBX Best Statistical Fit: '{best_cctbx_fit['bravais_type']}'")
    print(f"  Statistical Fit Score: {best_cctbx_fit['score_max_delta_deg']:.5f} degrees")
    print("\n  STATISTICAL VERDICT:")
    
    # Interpret the CCTBX results
    highest_symm_fit = None
    # The list is sorted by score, but we want the one with highest symmetry
    # that still has a very low score.
    # We check for the highest symmetry lattices first.
    # **FIX 14: Initialized the symmetry order list**
    # (These strings MUST match the 'bravais_type' output from CCTBX)
    symm_order = ['Cubic', 'Hexagonal', 'Rhombohedral', 'Tetragonal', 'Orthorhombic', 'Monoclinic', 'Triclinic']
    all_fits_by_type = {fit['bravais_type']: fit for fit in analysis['cctbx_fits']}
    
    for lattice_type in symm_order:
        if lattice_type in all_fits_by_type:
            # This is the highest symmetry type that fit within the tolerance
            highest_symm_fit = all_fits_by_type[lattice_type]
            break
            
    if highest_symm_fit:
        print(f"  The refined unit cell fits the constraints for the")
        print(f"  '{highest_symm_fit['bravais_type']}' Bravais lattice with a maximum angular")
        print(f"  deviation (fit score) of only {highest_symm_fit['score_max_delta_deg']:.5f} degrees.")
        print(f"\n  The most probable Bravais lattice is: {highest_symm_fit['bravais_type']}")
    else:
        # This should be rare, as Triclinic (aP) will always fit
        print("  Could not determine a high-symmetry Bravais lattice within tolerance.")

    print("======================================================")
    
    # Clean up temporary files
    if os.path.exists("spots.dat"): os.remove("spots.dat")
    if os.path.exists("spots.cor"): os.remove("spots.cor")


if __name__ == "__main__":
    main()