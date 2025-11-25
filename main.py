import numpy as np
from scipy.spatial import KDTree  # Still useful, but we'll use it in 3D


def convert_to_normals(points, L):
    """
    Converts detector (x, y) coordinates to 3D plane normal vectors.

    The incident beam S0 is (0, 0, 1).
    The diffracted beam S is proportional to (x, y, L).
    The plane normal N is proportional to (S_unit - S0_unit).

    Args:
        points (np.array): An (N, 2) array of (x, y) coordinates.
        L (float): The crystal-to-detector distance.

    Returns:
        np.array: An (M, 3) array of 3D normal vectors.
    """
    normal_vectors = []

    # S0_unit is the incident beam direction
    s0_unit = np.array([0.0, 0.0, 1.0])

    for x, y, z in points:
        R_squared = x**2 + y**2
        if R_squared < 1e-9:
            # Skip the center point (R=0)
            continue

        V_mag = np.sqrt(x**2 + y**2 + L**2)

        # S_unit is the diffracted beam direction
        s_unit = np.array([x / V_mag, y / V_mag, L / V_mag])

        # The normal 'g' is proportional to the difference
        g_vec = s_unit - s0_unit

        # Normalize the normal vector
        g_mag = np.linalg.norm(g_vec)
        if g_mag > 1e-9:
            normal_vectors.append(g_vec / g_mag)

    return np.array(normal_vectors)


def get_indexing_error(A, normals, max_index=10):
    """
    Calculates the 'badness' of an orientation matrix A.
    It applies A to all normals and checks how far the resulting
    (h,k,l) vectors are from small integers.
    """
    total_error = 0
    indexed_hkl = []

    for g_vec in normals:
        # Transform lab-frame normal g_vec to crystal-frame (h,k,l)
        hkl_raw = A @ g_vec

        # --- Updated Normalization Logic ---
        abs_hkl = np.abs(hkl_raw)
        # Find all components that are not effectively zero
        non_zero_indices = np.where(abs_hkl > 1e-5)[0]

        if len(non_zero_indices) == 0:
            indexed_hkl.append((0, 0, 0))  # Could not index
            continue

        # Find the index of the component with the smallest absolute value
        min_abs_val_idx_in_nonzero = np.argmin(abs_hkl[non_zero_indices])
        actual_idx = non_zero_indices[min_abs_val_idx_in_nonzero]

        # Get the *actual value* (with sign) at that index
        val_to_divide = hkl_raw[actual_idx]

        # Safety check (should be redundant, but good practice)
        if abs(val_to_divide) < 1e-9:
            indexed_hkl.append((0, 0, 0))
            continue

        # Normalize by this value to get (hopefully) integers
        hkl_norm = hkl_raw / val_to_divide
        # --- End of Updated Logic ---

        # Round to nearest integers
        hkl_int = np.round(hkl_norm).astype(int)

        # Check for large indices (bad fit)
        if np.any(np.abs(hkl_int) > max_index):
            total_error += 10  # Penalize heavily
            indexed_hkl.append(None)
            continue

        # The error is the "distance" from the integer
        error = np.sum((hkl_norm - hkl_int) ** 2)
        total_error += error
        indexed_hkl.append(tuple(hkl_int))

    return total_error, indexed_hkl


def find_cubic_orientation(normals):
    """
    Attempts to find the best orientation matrix by "seeding" two
    spots as (100) and (010).
    """
    best_fit = (np.inf, None, None)  # (error, A, hkl_list)
    num_normals = len(normals)

    if num_normals < 3:
        print("Error: Need at least 3 spots to determine orientation.")
        return None, None

    print(f"--- Searching for Cubic Orientation ---")
    print(f"Testing {num_normals * (num_normals - 1)} seed pairs...")

    # Loop through every pair of normals as seeds
    for i in range(num_normals):
        for j in range(num_normals):
            if i == j:
                continue

            # --- Seed Guess ---
            # g1_L = Lab frame vector for spot i, which we *guess* is (1,0,0)
            # g2_L = Lab frame vector for spot j, which we *guess* is (0,1,0)
            g1_L = normals[i]
            g2_L = normals[j]

            g1_C = np.array([1.0, 0.0, 0.0])
            g2_C = np.array([0.0, 1.0, 0.0])

            # We need a third vector to build a basis
            # g3_L = g1_L x g2_L
            # g3_C = g1_C x g2_C = (0,0,1)
            g3_L = np.cross(g1_L, g2_L)
            g3_C = np.array([0.0, 0.0, 1.0])

            # Create the Lab-frame matrix [g1, g2, g3]
            L_matrix = np.array([g1_L, g2_L, g3_L]).T  # As columns
            # Create the Crystal-frame matrix [g1, g2, g3]
            C_matrix = np.array([g1_C, g2_C, g3_C]).T  # As columns

            try:
                # We want A where C = A @ L
                # So, A = C @ L_inverse
                A = C_matrix @ np.linalg.inv(L_matrix)
            except np.linalg.LinAlgError:
                continue  # This pair was a bad seed (collinear vectors)

            # --- Score the Guess ---
            error, hkl_list = get_indexing_error(A, normals)

            if error < best_fit[0]:
                best_fit = (error, A, hkl_list)

    print(f"Best fit found with total error: {best_fit[0]:.2f}")
    return best_fit[1], best_fit[2]  # Return A, hkl_list


def analyze_bravais_lattice(hkl_list):
    """
    Performs the statistical check on a list of (h,k,l) tuples
    to determine the most likely Bravais lattice (SC, BCC, FCC, Diamond).
    """
    # Filter out un-indexed spots (None) or (0,0,0)
    reflections = [hkl for hkl in hkl_list if hkl and hkl != (0, 0, 0)]
    num_spots = len(reflections)

    if num_spots < 5:
        print("\nError: Could not confidently index enough spots.")
        print("Aborting Bravais lattice analysis.")
        return

    print(f"\n--- Analyzing {num_spots} Indexed Reflections for Cubic Lattices ---")

    sc_count = num_spots  # Simple Cubic allows all reflections
    bcc_count = 0
    fcc_count = 0
    diamond_count = 0

    for h, k, l in reflections:
        # Check for BCC: h+k+l must be even
        if (h + k + l) % 2 == 0:
            bcc_count += 1

        # Check for FCC: h,k,l must be all even or all odd
        all_even = (h % 2 == 0) and (k % 2 == 0) and (l % 2 == 0)
        all_odd = (h % 2 != 0) and (k % 2 != 0) and (l % 2 != 0)
        is_fcc = all_even or all_odd

        if is_fcc:
            fcc_count += 1

        # Check for Diamond Cubic: (a subset of FCC)
        # Rule: Must be all odd, OR all even AND (h+k+l) is a multiple of 4.
        # e.g., (2,2,0) is FCC, but (2+2+0)=4, so it's also Diamond.
        # e.g., (2,0,0) is FCC, but (2+0+0)=2, so it's NOT Diamond.
        if all_odd:
            diamond_count += 1
        elif all_even and (h + k + l) % 4 == 0:
            diamond_count += 1

    # Calculate statistics
    percent_sc = (sc_count / num_spots) * 100
    percent_bcc = (bcc_count / num_spots) * 100
    percent_fcc = (fcc_count / num_spots) * 100
    percent_diamond = (diamond_count / num_spots) * 100

    # Simple Cubic (SC) allows all reflections, so its
    # "fit" is always 100%, but it's the *least* restrictive.

    print(
        f"SC rule (all allowed) matched:      {sc_count}/{num_spots} ({percent_sc:.1f}%)"
    )
    print(
        f"BCC rule (h+k+l=even) matched:    {bcc_count}/{num_spots} ({percent_bcc:.1f}%)"
    )
    print(
        f"FCC rule (all even/odd) matched: {fcc_count}/{num_spots} ({percent_fcc:.1f}%)"
    )
    print(
        f"Diamond rule (FCC + h+k+l=4n) matched: {diamond_count}/{num_spots} ({percent_diamond:.1f}%)"
    )

    print("\n--- Conclusion ---")
    # We look for the MOST RESTRICTIVE rule that fits > 95%
    if percent_diamond > 95:
        print("The data strongly fits the selection rules for a")
        print("DIAMOND CUBIC lattice (e.g., Silicon, Germanium).")
    elif percent_fcc > 95:
        print("The data strongly fits the selection rules for a")
        print("FACE-CENTERED CUBIC (FCC) lattice (e.g., Aluminum, Copper).")
    elif percent_bcc > 95:
        print("The data strongly fits the selection rules for a")
        print("BODY-CENTERED CUBIC (BCC) lattice (e.g., Iron, Tungsten).")
    # If no specific rule fits, it's likely Simple Cubic
    elif percent_bcc < 60 and percent_fcc < 40 and percent_diamond < 30:
        # These thresholds are estimates for a random-like distribution
        print("The data does not fit BCC, FCC, or Diamond rules.")
        print("This strongly suggests a SIMPLE CUBIC (SC) lattice.")
    else:
        print("The results are ambiguous.")
        print("The fit is not high enough for a conclusive result,")
        print("which could mean the crystal is not Cubic,")
        print("the indexing was incorrect, or the data is noisy.")


# --- Your Raw Data ---
data = [
    (-2.350074962518741, -32.89036544850499, 3.581571132950927),
    (-11.915292353823089, -28.03156146179402, 3.0612381691178143),
    (7.050224887556222, -27.325581395348838, 2.6315598739623454),
    (-1.0307346326836582, -19.102990033222593, 1.2150344446358758),
    (11.049475262368816, -19.102990033222593, 1.614692994389884),
    (-14.760119940029986, -18.646179401993358, 1.8734379242644081),
    (3.0922038980509745, -17.691029900332225, 1.071288681475039),
    (-5.607196401799101, -17.524916943521596, 1.1243308185868557),
    (8.905547226386807, -17.19269102990033, 1.2444954246303155),
    (-11.79160419790105, -16.777408637873755, 1.395255441384819),
    (1.3193403298350825, -14.950166112956811, 0.7489572956010022),
    (-3.0922038980509745, -14.78405315614618, 0.7585153570798298),
    (6.060719640179911, -14.41029900332226, 0.8124299914367725),
    (-8.287106446776612, -14.03654485049834, 0.8830697083018038),
    (17.027736131934034, -13.45514950166113, 1.5618185622316503),
    (10.967016491754123, -12.70764119601329, 0.9362766053841085),
    (-6.101949025487257, -11.918604651162791, 0.5964372710734551),
    (-21.274362818590706, -11.544850498338871, 1.9403899111952114),
    (-13.853073463268366, -11.461794019933555, 1.0737580340612851),
    (15.254872563718141, -10.880398671096346, 1.1657838671743832),
    (-0.5772113943028486, -10.838870431893689, 0.3922015439400752),
    (24.655172413793103, -10.672757475083056, 2.386959018400944),
    (9.482758620689657, -10.589700996677742, 0.6720427891561371),
    (-11.58545727136432, -9.883720930232558, 0.7710541172055514),
    (2.721139430284858, -9.67607973421927, 0.33639319480170116),
    (-3.7106446776611697, -9.426910299003323, 0.34172914300711454),
    (-18.800599700149927, -8.970099667774086, 1.4395101587927002),
    (7.586206896551724, -8.596345514950166, 0.43752088920402343),
    (13.193403298350825, -7.848837209302326, 0.7835207711076748),
    (-8.86431784107946, -7.641196013289036, 0.4558540147281178),
    (4.411544227886057, -7.350498338870432, 0.2447721163845813),
    (-5.236131934032984, -6.976744186046512, 0.25342604103286703),
    (-15.254872563718141, -5.897009966777409, 0.8889852291530076),
    (11.58545727136432, -5.897009966777409, 0.5622713256374823),
    (1.1131934032983508, -5.813953488372094, 0.11675874038215284),
    (-1.360569715142429, -5.7724252491694354, 0.11719436229486746),
    (16.49175412293853, -5.647840531561462, 1.0095230663331733),
    (4.94752623688156, -5.274086378737541, 0.17421217704128367),
    (6.844077961019491, -5.107973421926911, 0.24291263023232545),
    (-5.4010494752623694, -4.775747508305648, 0.17316371341891568),
    (-7.586206896551724, -4.235880398671097, 0.251433330365046),
    (-12.616191904047977, -3.903654485049834, 0.580233817385249),
    (9.152923538230885, -3.737541528239203, 0.32546432980677764),
    (14.100449775112445, -3.280730897009967, 0.6970002325164728),
    (-18.63568215892054, -2.574750830564784, 1.1751235864148555),
    (-9.936281859070466, -2.200996677740864, 0.34485053888050743),
    (18.223388305847077, -1.5780730897009967, 1.1111584100331129),
    (5.442278860569716, -1.536544850498339, 0.10656004743361791),
    (10.595952023988007, -0.872093023255814, 0.3763104532621071),
    (-5.73088455772114, -0.49833887043189373, 0.11026407092791146),
    (-15.089955022488757, -0.4568106312292359, 0.7578038395808449),
    (5.73088455772114, 0.4568106312292359, 0.1101319490684034),
    (14.47151424287856, 0.6644518272425249, 0.6979303796590557),
    (-10.6784107946027, 1.287375415282392, 0.3851249045535212),
    (-5.524737631184408, 2.0348837209302326, 0.11550045765136474),
    (-19.583958020989506, 2.159468438538206, 1.2884487186808258),
    (9.730134932533733, 2.200996677740864, 0.3313670269140516),
    (17.31634182908546, 2.574750830564784, 1.0181612793018644),
    (7.668665667166417, 3.8621262458471763, 0.24554719609375297),
    (-14.677661169415293, 3.903654485049834, 0.7669468276876898),
    (12.57496251874063, 3.9451827242524917, 0.5778673944994921),
    (5.689655172413794, 4.485049833887043, 0.17485757607173014),
    (-8.9880059970015, 4.568106312292359, 0.3384576450168879),
    (2.1851574212893556, 5.523255813953488, 0.11755815940446723),
    (14.677661169415293, 5.5647840531561465, 0.8190987871311393),
    (-6.2256371814092955, 5.855481727574751, 0.2432867870525115),
    (-3.999250374812594, 5.897009966777409, 0.1691337462819149),
    (0.08245877061469266, 5.980066445182724, 0.1191793014392033),
    (5.648425787106447, 6.478405315614618, 0.24604636830159166),
    (-17.481259370314845, 6.810631229235881, 1.1687108065450786),
    (-11.66791604197901, 7.01827242524917, 0.616720229188644),
    (9.070464767616192, 7.142857142857143, 0.4436563609906159),
    (-3.4220389805097455, 7.931893687707642, 0.2485450453921203),
    (17.893553223388306, 8.222591362126247, 1.2871119946003375),
    (4.535232383808096, 9.09468438538206, 0.3438778828218574),
    (-13.317091454272864, 9.136212624584719, 0.8668794199810179),
    (-6.844077961019491, 9.30232558139535, 0.44392531557662096),
    (11.58545727136432, 9.385382059800666, 0.7392059704277472),
    (-1.607946026986507, 9.925249169435217, 0.3366091858569007),
    (19.707646176911545, 10.340531561461795, 1.6420717044182425),
    (1.6904047976011995, 10.672757475083056, 0.38871374043230844),
    (13.60569715142429, 10.79734219269103, 1.0023098942606907),
    (7.008995502248876, 11.37873754152824, 0.5941621909346395),
    (-8.905547226386807, 11.877076411960132, 0.7327891186816373),
    (-16.120689655172416, 12.832225913621263, 1.40852900961886),
    (-3.2983508245877062, 13.205980066445184, 0.6163239083911662),
    (8.905547226386807, 13.330564784053157, 0.8542764685948327),
    (-28.035982008995504, 13.579734219269103, 3.2006053143221322),
    (-10.224887556221889, 13.953488372093023, 0.9941991047587067),
    (4.370314842578711, 14.285714285714286, 0.742101897500163),
    (-0.20614692653673164, 14.908637873754154, 0.7392118189758037),
    (-17.646176911544227, 15.448504983388705, 1.8224089711901001),
    (-4.988755622188906, 15.490033222591363, 0.8801803150264789),
    (12.286356821589207, 15.780730897009967, 1.3274133512869355),
    (6.926536731634183, 17.109634551495017, 1.1314543884905106),
    (15.007496251874064, 17.400332225913623, 1.749782554447762),
    (-1.8553223388305848, 18.147840531561464, 1.1052161142690409),
    (-7.833583208395803, 18.521594684385384, 1.3420447051496467),
    (2.514992503748126, 19.102990033222593, 1.2324350643846174),
    (-9.977511244377812, 20.971760797342192, 1.7872375451000266),
    (12.739880059970016, 27.11794019933555, 2.9630256780937145),
    (-5.442278860569716, 29.56810631229236, 2.983304023972579),
    (4.370314842578711, 33.55481727574751, 3.7693903682786356),
]


# --- Run the Analysis ---
if __name__ == "__main__":
    print("NOTE: This script assumes a CUBIC crystal system for indexing.")
    print(
        "It will analyze the indexed (h,k,l) spots for SC, BCC, FCC, and Diamond Cubic rules.\n"
    )

    L_distance = 150  # Crystal-to-detector distance in mm
    if not data:
        print("Error: The data list is empty. Please add your (x, y) coordinates.")
    else:
        # 1. Center the data
        points = np.array(data)
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        print(f"Data centroid calculated at: ({centroid[0]:.3f}, {centroid[1]:.3f})")

        # 2. Convert to 3D normals
        normals = convert_to_normals(centered_points, L_distance)
        print(f"Converted {len(normals)} spots to 3D normal vectors.")

        # 3. Try to index the pattern (assuming Cubic)
        A, hkl_list = find_cubic_orientation(normals)

        if hkl_list:
            # 4. Analyze the indexed (h,k,l) list for Bravais rules
            analyze_bravais_lattice(hkl_list)
        else:
            print("Could not find a good indexing solution.")
            print("The crystal may not be Cubic, or the data is too noisy.")
