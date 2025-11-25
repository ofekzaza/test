raw_data = [
    (-2.350074962518741, -32.89036544850499, 3.581571132950927), (-11.915292353823089, -28.03156146179402, 3.0612381691178143),
    (7.050224887556222, -27.325581395348838, 2.6315598739623454), (-1.0307346326836582, -19.102990033222593, 1.2150344446358758),
    (11.049475262368816, -19.102990033222593, 1.614692994389884), (-14.760119940029986, -18.646179401993358, 1.8734379242644081),
    (3.0922038980509745, -17.691029900332225, 1.071288681475039), (-5.607196401799101, -17.524916943521596, 1.1243308185868557),
    (8.905547226386807, -17.19269102990033, 1.2444954246303155), (-11.79160419790105, -16.777408637873755, 1.395255441384819),
    (1.3193403298350825, -14.950166112956811, 0.7489572956010022), (-3.0922038980509745, -14.78405315614618, 0.7585153570798298),
    (6.060719640179911, -14.41029900332226, 0.8124299914367725), (-8.287106446776612, -14.03654485049834, 0.8830697083018038),
    (17.027736131934034, -13.45514950166113, 1.5618185622316503), (10.967016491754123, -12.70764119601329, 0.9362766053841085),
    (-6.101949025487257, -11.918604651162791, 0.5964372710734551), (-21.274362818590706, -11.544850498338871, 1.9403899111952114),
    (-13.853073463268366, -11.461794019933555, 1.0737580340612851), (15.254872563718141, -10.880398671096346, 1.1657838671743832),
    (-0.5772113943028486, -10.838870431893689, 0.3922015439400752), (24.655172413793103, -10.672757475083056, 2.386959018400944),
    (9.482758620689657, -10.589700996677742, 0.6720427891561371), (-11.58545727136432, -9.883720930232558, 0.7710541172055514),
    (2.721139430284858, -9.67607973421927, 0.33639319480170116), (-3.7106446776611697, -9.426910299003323, 0.34172914300711454),
    (-18.800599700149927, -8.970099667774086, 1.4395101587927002), (7.586206896551724, -8.596345514950166, 0.43752088920402343),
    (13.193403298350825, -7.848837209302326, 0.7835207711076748), (-8.86431784107946, -7.641196013289036, 0.4558540147281178),
    (4.411544227886057, -7.350498338870432, 0.2447721163845813), (-5.236131934032984, -6.976744186046512, 0.25342604103286703),
    (-15.254872563718141, -5.897009966777409, 0.8889852291530076), (11.58545727136432, -5.897009966777409, 0.5622713256374823),
    (1.1131934032983508, -5.813953488372094, 0.11675874038215284), (-1.360569715142429, -5.7724252491694354, 0.11719436229486746),
    (16.49175412293853, -5.647840531561462, 1.0095230663331733), (4.94752623688156, -5.274086378737541, 0.17421217704128367),
    (6.844077961019491, -5.107973421926911, 0.24291263023232545), (-5.4010494752623694, -4.775747508305648, 0.17316371341891568),
    (-7.586206896551724, -4.235880398671097, 0.251433330365046), (-12.616191904047977, -3.903654485049834, 0.580233817385249),
    (9.152923538230885, -3.737541528239203, 0.32546432980677764), (14.100449775112445, -3.280730897009967, 0.6970002325164728),
    (-18.63568215892054, -2.574750830564784, 1.1751235864148555), (-9.936281859070466, -2.200996677740864, 0.34485053888050743),
    (18.223388305847077, -1.5780730897009967, 1.1111584100331129), (5.442278860569716, -1.536544850498339, 0.10656004743361791),
    (10.595952023988007, -0.872093023255814, 0.3763104532621071), (-5.73088455772114, -0.49833887043189373, 0.11026407092791146),
    (-15.089955022488757, -0.4568106312292359, 0.7578038395808449), (5.73088455772114, 0.4568106312292359, 0.1101319490684034),
    (14.47151424287856, 0.6644518272425249, 0.6979303796590557), (-10.6784107946027, 1.287375415282392, 0.3851249045535212),
    (-5.524737631184408, 2.0348837209302326, 0.11550045765136474), (-19.583958020989506, 2.159468438538206, 1.2884487186808258),
    (9.730134932533733, 2.200996677740864, 0.3313670269140516), (17.31634182908546, 2.574750830564784, 1.0181612793018644),
    (7.668665667166417, 3.8621262458471763, 0.24554719609375297), (-14.677661169415293, 3.903654485049834, 0.7669468276876898),
    (12.57496251874063, 3.9451827242524917, 0.5778673944994921), (5.689655172413794, 4.485049833887043, 0.17485757607173014),
    (-8.9880059970015, 4.568106312292359, 0.3384576450168879), (2.1851574212893556, 5.523255813953488, 0.11755815940446723),
    (14.677661169415293, 5.5647840531561465, 0.8190987871311393), (-6.2256371814092955, 5.855481727574751, 0.2432867870525115),
    (-3.999250374812594, 5.897009966777409, 0.1691337462819149), (0.08245877061469266, 5.980066445182724, 0.1191793014392033),
    (5.648425787106447, 6.478405315614618, 0.24604636830159166), (-17.481259370314845, 6.810631229235881, 1.1687108065450786),
    (-11.66791604197901, 7.01827242524917, 0.616720229188644), (9.070464767616192, 7.142857142857143, 0.4436563609906159),
    (-3.4220389805097455, 7.931893687707642, 0.2485450453921203), (17.893553223388306, 8.222591362126247, 1.2871119946003375),
    (4.535232383808096, 9.09468438538206, 0.3438778828218574), (-13.317091454272864, 9.136212624584719, 0.8668794199810179),
    (-6.844077961019491, 9.30232558139535, 0.44392531557662096), (11.58545727136432, 9.385382059800666, 0.7392059704277472),
    (-1.607946026986507, 9.925249169435217, 0.3366091858569007), (19.707646176911545, 10.340531561461795, 1.6420717044182425),
    (1.6904047976011995, 10.672757475083056, 0.38871374043230844), (13.60569715142429, 10.79734219269103, 1.0023098942606907),
    (7.008995502248876, 11.37873754152824, 0.5941621909346395), (-8.905547226386807, 11.877076411960132, 0.7327891186816373),
    (-16.120689655172416, 12.832225913621263, 1.40852900961886), (-3.2983508245877062, 13.205980066445184, 0.6163239083911662),
    (8.905547226386807, 13.330564784053157, 0.8542764685948327), (-28.035982008995504, 13.579734219269103, 3.2006053143221322),
    (-10.224887556221889, 13.953488372093023, 0.9941991047587067), (4.370314842578711, 14.285714285714286, 0.742101897500163),
    (-0.20614692653673164, 14.908637873754154, 0.7392118189758037), (-17.646176911544227, 15.448504983388705, 1.8224089711901001),
    (-4.988755622188906, 15.490033222591363, 0.8801803150264789), (12.286356821589207, 15.780730897009967, 1.3274133512869355),
    (6.926536731634183, 17.109634551495017, 1.1314543884905106), (15.007496251874064, 17.400332225913623, 1.749782554447762),
    (-1.8553223388305848, 18.147840531561464, 1.1052161142690409), (-7.833583208395803, 18.521594684385384, 1.3420447051496467),
    (2.514992503748126, 19.102990033222593, 1.2324350643846174), (-9.977511244377812, 20.971760797342192, 1.7872375451000266),
    (12.739880059970016, 27.11794019933555, 2.9630256780937145), (-5.442278860569716, 29.56810631229236, 2.983304023972579),
    (4.370314842578711, 33.55481727574751, 3.7693903682786356)
]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def find_reciprocal_pairs_in_subset(points_subset, transform_func, threshold):
    """
    Finds "mutual best match" pairs within a subset of points.
    
    A pair (i, j) is formed if:
    1. j is the closest point to transform_func(i)
    2. i is the closest point to transform_func(j)
    3. The distance for (1) is below the threshold.
    
    Args:
        points_subset (np.ndarray): (M, N) array of M points in N dimensions.
        transform_func (callable): A function that takes an (N,) point
                                   or (M, N) array and returns the
                                   ideal symmetric partner(s).
        threshold (float): The maximum distance to be considered a pair.
                           
    Returns:
        np.ndarray: An array of length M. 
                    map[i] = pair_id (0, 1, 2...) if paired.
                    map[i] = -1 if unpaired (orphan).
    """
    n_points = len(points_subset)
    pair_map = np.full(n_points, -1, dtype=int)
    
    unpaired_indices = set(range(n_points))
    pair_id_counter = 0
    
    # Pre-calculate all ideal partners
    ideal_partners = transform_func(points_subset)
    
    while unpaired_indices:
        i_local = unpaired_indices.pop()
        
        # --- Check for i's best match ---
        search_indices_i = list(unpaired_indices)
        if not search_indices_i:
            break
            
        candidates_i = points_subset[search_indices_i]
        
        # Find distance from i's ideal partner to all candidates
        dists_sq_i = np.sum((candidates_i - ideal_partners[i_local])**2, axis=1)
        
        min_idx_in_search = np.argmin(dists_sq_i)
        min_dist = np.sqrt(dists_sq_i[min_idx_in_search])
        
        # This is the global index (from 0 to n_points-1)
        j_local = search_indices_i[min_idx_in_search] 
        
        # --- Check for j's best match (Reciprocity) ---
        search_indices_j = list(unpaired_indices.union({i_local}))
        search_indices_j.remove(j_local)
        
        candidates_j = points_subset[search_indices_j]
        
        # Find distance from j's ideal partner to all candidates (incl. i)
        dists_sq_j = np.sum((candidates_j - ideal_partners[j_local])**2, axis=1)
        
        best_match_for_j_idx = search_indices_j[np.argmin(dists_sq_j)]
        
        # --- Check all conditions ---
        if (min_dist < threshold) and (best_match_for_j_idx == i_local):
            # It's a mutual best match!
            unpaired_indices.remove(j_local)
            pair_map[i_local] = pair_id_counter
            pair_map[j_local] = pair_id_counter
            pair_id_counter += 1
            
    return pair_map

def plot_results(title, points_original, main_pair_map):
    """Helper function to generate the plot"""
    total_pairs_found = len(np.unique(main_pair_map[main_pair_map != -1]))
    num_orphans_final = np.sum(main_pair_map == -1)
    
    print(f"\n--- FINAL TALLY for {title} ---")
    print(f"Total pairs:     {total_pairs_found}")
    print(f"Final orphans:   {num_orphans_final}")
    print(f"Generating plot for {title}...")

    plt.figure(figsize=(16, 16))
    
    colors = plt.cm.get_cmap('hsv', total_pairs_found) if total_pairs_found > 0 else None
        
    for i in range(len(points_original)):
        point = points_original[i]
        pair_id = main_pair_map[i]
        
        if pair_id == -1:
            plt.scatter(point[0], point[1], c='black', marker='x', s=100)
            label = "Orphan"
        else:
            color = colors(pair_id / total_pairs_found)
            plt.scatter(point[0], point[1], c=[color], s=50)
            label = str(pair_id)

        plt.text(point[0] + 0.1, point[1] + 0.1, label, 
                 fontsize=9, ha='left', va='bottom')

    plt.title(f'{title} ({total_pairs_found} Pairs, {num_orphans_final} Orphans)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def run_symmetry_analysis(title, points_original, points_centered, 
                          transform_func, runs_config):
    """
    Runs the full multi-pass analysis for a given symmetry.
    
    Args:
        title (str): Title for the analysis and plot.
        points_original (np.ndarray): Original (N, 3) data for plotting.
        points_centered (np.ndarray): Centered (N, 3) data for calculation.
        transform_func (callable): The symmetry operation function.
        runs_config (list): List of (dims, threshold) tuples.
    """
    print(f"\n=============================================")
    print(f"--- Starting Analysis for: {title} ---")
    print(f"=============================================")
    
    main_pair_map = np.full(len(points_original), -1, dtype=int)
    global_pair_id_counter = 0
    total_pairs_found = 0

    for i, (dims, threshold) in enumerate(runs_config):
        run_num = i + 1
        
        orphan_indices = np.where(main_pair_map == -1)[0]
        
        if len(orphan_indices) < 2:
            print(f"--- Run {run_num} skipped (not enough orphans left) ---")
            break
        
        # Get data for this run, using specified dimensions
        run_data = points_centered[orphan_indices, :dims]
        
        # Create a transform function that only operates on the correct dims
        transform_lambda = lambda p: transform_func(p, dims)
        
        # Find pairs *within this subset*
        run_pair_map = find_reciprocal_pairs_in_subset(
            run_data, 
            transform_lambda, 
            threshold
        )
        
        # --- Merge results back into the main map ---
        num_pairs_in_run = 0
        unique_run_ids = np.unique(run_pair_map[run_pair_map != -1])
        
        for local_id in unique_run_ids:
            local_indices = np.where(run_pair_map == local_id)[0]
            
            global_idx_1 = orphan_indices[local_indices[0]]
            global_idx_2 = orphan_indices[local_indices[1]]
            
            main_pair_map[global_idx_1] = global_pair_id_counter
            main_pair_map[global_idx_2] = global_pair_id_counter
            
            global_pair_id_counter += 1
            num_pairs_in_run += 1
        
        print(f"--- Run {run_num} (Dims={dims}, Thresh={threshold}) ---")
        print(f"    Found {num_pairs_in_run} new pairs.")
        print(f"    Total pairs so far: {global_pair_id_counter}")
        total_pairs_found += num_pairs_in_run

    # --- Plot the final results for this symmetry ---
    plot_results(title, points_original, main_pair_map)


# --- 1. Define Symmetry Transformations ---
# These functions must handle N-dimensional slices

def transform_inversion(p, dims):
    """Symmetry 1: (x, y, z) -> (-x, -y, -z)"""
    return -p[..., :dims]

def transform_y_mirror(p, dims):
    """Symmetry 2: (x, y, z) -> (x, -y, z)"""
    p_out = p[..., :dims].copy()
    if dims > 1:
        p_out[..., 1] = -p_out[..., 1] # Invert Y
    return p_out

def transform_x_mirror(p, dims):
    """Symmetry 3: (x, y, z) -> (-x, y, z)"""
    p_out = p[..., :dims].copy()
    if dims > 0:
        p_out[..., 0] = -p_out[..., 0] # Invert X
    return p_out


# --- 2. Load Your Data ---
# --- Inject your raw_data list here ---
# --------------------------------------

points_original = np.array(raw_data)

if points_original.size == 0:
    print("Please inject your data into the 'raw_data' list and re-run.")
else:
    # --- 3. Center the Data (once) ---
    center = np.mean(points_original, axis=0)
    points_centered = points_original - center
    print(f"Data Loaded. Calculated center of symmetry: {center}")

    # --- 4. Define the Multi-Pass Runs ---
    # (dims_to_use, threshold)
    # *** TUNE THESE RUNS ***
    RUNS_CONFIG = [
    (3, 1),  # Run 1: Strict 3D match (high quality)
    (3, 1.5),  # Run 2: Looser 3D match (medium quality)
    (3, 2),   # Run 3: Loose 2D match (your 'r' idea, low quality
    (3, 2.5),   # Run 3: Loose 2D match (your 'r' idea, low quality)
    (3,3),
    (3, 3.5),
    (3, 4),
    (3, 4.5),
    (3, 5),
    (3, 5.5),
    (3, 6),

    ]

    # --- 5. Execute Analysis for Each Symmetry ---
    
    # Run 1: Inversion Symmetry
    run_symmetry_analysis(
        title="Symmetry 1: Inversion",
        points_original=points_original,
        points_centered=points_centered,
        transform_func=transform_inversion,
        runs_config=RUNS_CONFIG
    )
    
    # Run 2: Y-Mirror (across XZ plane)
    run_symmetry_analysis(
        title="Symmetry 2: Y-Mirror (across XZ plane)",
        points_original=points_original,
        points_centered=points_centered,
        transform_func=transform_y_mirror,
        runs_config=RUNS_CONFIG
    )
    
    # Run 3: X-Mirror (across YZ plane)
    run_symmetry_analysis(
        title="Symmetry 3: X-Mirror (across YZ plane)",
        points_original=points_original,
        points_centered=points_centered,
        transform_func=transform_x_mirror,
        runs_config=RUNS_CONFIG
    )