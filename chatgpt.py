#!/usr/bin/env python3
"""
Laue Bravais Lattice Fitter
---------------------------
Given 2D detector spot coordinates (x, y) [mm] from a Laue experiment and the
sample–detector distance L [mm], this script attempts to fit the pattern to each
of the 14 Bravais lattice types and ranks them statistically by a robust score.

Model (simplified, yet effective for Laue indexing pre-fit):
- Assume the diffracted directions s' are parallel to reciprocal-lattice
  directions R * (B* @ h), where B* are reciprocal basis vectors from a trial
  unit cell and R is an orientation matrix.
- Project unit vectors u = v / ||v|| onto a flat detector plane at z = L using
  a pinhole camera model: (x, y) = L * (u_x/u_z, u_y/u_z) with u_z>0.
- Generate a bank of candidate reflections h = (h,k,l) within |h|,|k|,|l| ≤ Hmax
  satisfying the centering extinction rules of the Bravais type.
- For a given parameter set, match predicted spots to measured ones using a
  KD-tree within a capture radius (tol), compute RMSE over inliers, and a robust
  score that balances low error and high coverage.

Bravais lattices covered (14):
  1. cubic_P      (cP)
  2. cubic_I      (cI)
  3. cubic_F      (cF)
  4. tetragonal_P (tP)
  5. tetragonal_I (tI)
  6. orthorhombic_P (oP)
  7. orthorhombic_C (oC)
  8. orthorhombic_I (oI)
  9. orthorhombic_F (oF)
 10. monoclinic_P (mP)
 11. monoclinic_C (mC) [unique b setting]
 12. rhombohedral_R (hR) [rhombohedral axes; reflection condition h+k+l=3n]
 13. hexagonal_P (hP)
 14. triclinic_P (aP)

USAGE:
- Just run the file: it uses the spots and L=15 mm embedded below.
- It will print a ranked table and save a small summary plot "fit_summary.png".

NOTE:
- This is a *search* problem; results depend on random restarts. Increase
  N_STARTS and generations for more robustness; it may take longer.
- The physics is simplified; for final indexing you would refine with full Laue
  equations (including wavelength bandpass, incident direction, detector tilt).

Requires: numpy, scipy, matplotlib
"""

import numpy as np
from numpy import sin, cos, tan, sqrt
from dataclasses import dataclass
from typing import Tuple, List, Dict
import itertools
import math

from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# -------------------- User Data --------------------
# Detector distance [mm]
L = 15.0

# Measured Laue spots [mm]
spots = np.array([
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
    (-8.19, -0.207), (-10.558,  1.285), (-19.19,  2.197), (9.234,  2.197),
    (16.661,  2.612), (-14.734,  2.902), (14.573,  3.607), (7.266,  3.897),
    (13.81,  4.27), (-8.912,  4.519), (14.091,  5.431), (-6.223,  5.763),
    (-13.73,  5.929), (5.38,  6.468), (-13.891,  6.551), (-17.102,  6.758),
    (8.672,  7.131), (-3.533,  7.877), (4.255,  8.955), (-13.088,  8.997),
    (11.321,  9.287), (-6.825,  9.328), (-1.726,  9.95), (19.029, 10.282),
    (12.847, 10.614), (1.445, 10.655), (-8.953, 11.94), (8.511, 13.267),
    (-10.478, 14.055), (3.854, 14.345), (-6.624, 14.884), (-0.201, 14.967),
    (-5.058, 15.257), (-17.383, 15.423), (6.343, 17.04), (14.292, 17.454),
    (-2.088, 18.242), (2.369, 19.03), (-10.036, 20.937)
], dtype=float)

# -------------------- Utilities --------------------

def deg2rad(x):
    return x * np.pi / 180.0

def rad2deg(x):
    return x * 180.0 / np.pi

# Build direct lattice vectors (a1,a2,a3) in Cartesian from (a,b,c,alpha,beta,gamma)
# Angles in degrees.
# See e.g. International Tables conventions.

def direct_lattice_vectors(a, b, c, alpha, beta, gamma):
    alpha_r, beta_r, gamma_r = map(deg2rad, (alpha, beta, gamma))
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * cos(gamma_r), b * sin(gamma_r), 0.0])
    cx = c * cos(beta_r)
    cy = c * (cos(alpha_r) - cos(beta_r) * cos(gamma_r)) / max(1e-12, sin(gamma_r))
    # Volume V = a · (b × c)
    vz_sq = c**2 - cx**2 - cy**2
    vz = math.sqrt(max(vz_sq, 1e-12))
    vc = np.array([cx, cy, vz])
    return va, vb, vc

# Reciprocal basis (omit 2π): b1 = (b x c)/V, etc.

def reciprocal_basis(va, vb, vc):
    V = np.dot(va, np.cross(vb, vc))
    b1 = np.cross(vb, vc) / V
    b2 = np.cross(vc, va) / V
    b3 = np.cross(va, vb) / V
    return b1, b2, b3

# Orientation matrix from ZXZ Euler angles (phi, theta, psi) [rad]

def R_ZXZ(phi, theta, psi):
    c1,s1 = np.cos(phi), np.sin(phi)
    c2,s2 = np.cos(theta), np.sin(theta)
    c3,s3 = np.cos(psi), np.sin(psi)
    Rz1 = np.array([[c1,-s1,0],[s1,c1,0],[0,0,1]])
    Rx2 = np.array([[1,0,0],[0,c2,-s2],[0,s2,c2]])
    Rz3 = np.array([[c3,-s3,0],[s3,c3,0],[0,0,1]])
    return Rz1 @ Rx2 @ Rz3

# Project a set of 3D unit vectors u (N,3) to detector plane z=L

def project_to_detector(u, L):
    # Only rays with positive z reach the detector in front of sample
    mask = u[:,2] > 1e-6
    u = u[mask]
    x = L * (u[:,0] / u[:,2])
    y = L * (u[:,1] / u[:,2])
    pts = np.stack([x,y], axis=1)
    return pts, mask

# Centering extinction rules

def allow_hkl(bravais: str, h: int, k: int, l: int) -> bool:
    if bravais.endswith('_P'):
        return True
    if bravais.endswith('_I'):
        return ((h + k + l) % 2) == 0
    if bravais.endswith('_F'):
        e = (h % 2 == 0, k % 2 == 0, l % 2 == 0)
        return all(e) or (not any(e))
    if bravais.endswith('_C'):
        # C-centred (0,1/2,1/2): k+l even
        return ((k + l) % 2) == 0
    if bravais == 'rhombohedral_R':
        # Rhombohedral axes: h+k+l = 3n
        return ((h + k + l) % 3) == 0
    # default allow
    return True

# Parameterizations and bounds per Bravais lattice
@dataclass
class LatticeParam:
    name: str
    # bounds for the parameters in the optimizer
    # We optimize: geometric cell params + orientation angles
    # Each returns (lower_bounds, upper_bounds) and a builder that maps params->(a,b,c,alpha,beta,gamma)
    
    def bounds(self) -> Tuple[List[float], List[float]]:
        raise NotImplementedError

    def build_cell(self, p: np.ndarray) -> Tuple[float,float,float,float,float,float]:
        raise NotImplementedError

class CubicP(LatticeParam):
    def __init__(self): self.name = 'cubic_P'
    def bounds(self):
        # a in [1, 5] arbitrary units; angles fixed 90
        return [1.0, 0.0, 0.0, 0.0], [5.0, 2*np.pi, np.pi, 2*np.pi]
    def build_cell(self, p):
        a = p[0]; return a,a,a,90,90,90

class CubicI(CubicP):
    def __init__(self): self.name = 'cubic_I'

class CubicF(CubicP):
    def __init__(self): self.name = 'cubic_F'

class TetragonalP(LatticeParam):
    def __init__(self): self.name = 'tetragonal_P'
    def bounds(self):
        return [1.0, 1.0, 0.0, 0.0, 0.0], [5.0, 5.0, 2*np.pi, np.pi, 2*np.pi]
    def build_cell(self, p):
        a,c = p[0], p[1]; return a,a,c,90,90,90

class TetragonalI(TetragonalP):
    def __init__(self): self.name = 'tetragonal_I'

class OrthorhombicP(LatticeParam):
    def __init__(self): self.name = 'orthorhombic_P'
    def bounds(self):
        return [1.0,1.0,1.0, 0.0,0.0,0.0], [5.0,5.0,5.0, 2*np.pi,np.pi,2*np.pi]
    def build_cell(self, p):
        a,b,c = p[:3]; return a,b,c,90,90,90

class OrthorhombicC(OrthorhombicP):
    def __init__(self): self.name = 'orthorhombic_C'
class OrthorhombicI(OrthorhombicP):
    def __init__(self): self.name = 'orthorhombic_I'
class OrthorhombicF(OrthorhombicP):
    def __init__(self): self.name = 'orthorhombic_F'

class MonoclinicP(LatticeParam):
    def __init__(self): self.name = 'monoclinic_P'
    def bounds(self):
        # unique b setting, beta variable
        return [1.0,1.0,1.0,  60.0*np.pi/180, 0.0,0.0], [5.0,5.0,5.0,  120.0*np.pi/180, 2*np.pi,np.pi]
    def build_cell(self, p):
        a,b,c = p[:3]
        beta = rad2deg(p[3])
        return a,b,c,90,beta,90

class MonoclinicC(MonoclinicP):
    def __init__(self): self.name = 'monoclinic_C'

class RhombohedralR(LatticeParam):
    def __init__(self): self.name = 'rhombohedral_R'
    def bounds(self):
        # a in [1,5], alpha in [60,120] deg (≠90)
        return [1.0, 60.0*np.pi/180, 0.0, 0.0, 0.0], [5.0, 120.0*np.pi/180, 2*np.pi, np.pi, 2*np.pi]
    def build_cell(self, p):
        a = p[0]; alpha = rad2deg(p[1])
        return a,a,a,alpha,alpha,alpha

class HexagonalP(LatticeParam):
    def __init__(self): self.name = 'hexagonal_P'
    def bounds(self):
        return [1.0,1.0, 0.0,0.0,0.0], [5.0,5.0, 2*np.pi,np.pi,2*np.pi]
    def build_cell(self, p):
        a,c = p[0], p[1]
        return a,a,c,90,90,120

class TriclinicP(LatticeParam):
    def __init__(self): self.name = 'triclinic_P'
    def bounds(self):
        return [1.0,1.0,1.0,  60.0*np.pi/180,60.0*np.pi/180,60.0*np.pi/180,  0.0,0.0,0.0], \
               [5.0,5.0,5.0, 120.0*np.pi/180,120.0*np.pi/180,120.0*np.pi/180, 2*np.pi,np.pi,2*np.pi]
    def build_cell(self, p):
        a,b,c = p[:3]
        alpha,beta,gamma = map(rad2deg, p[3:6])
        return a,b,c,alpha,beta,gamma

# Registry of the 14 types
LATTICES: List[LatticeParam] = [
    CubicP(), CubicI(), CubicF(),
    TetragonalP(), TetragonalI(),
    OrthorhombicP(), OrthorhombicC(), OrthorhombicI(), OrthorhombicF(),
    MonoclinicP(), MonoclinicC(),
    RhombohedralR(),
    HexagonalP(),
    TriclinicP(),
]

# -------------------- Prediction and Scoring --------------------

def make_predicted_points(bravais: str, cell_params: Tuple[float,float,float,float,float,float], angles: Tuple[float,float,float],
                          Hmax: int = 6, L: float = 15.0, max_points: int = 5000) -> np.ndarray:
    a,b,c,alpha,beta,gamma = cell_params
    va,vb,vc = direct_lattice_vectors(a,b,c,alpha,beta,gamma)
    b1,b2,b3 = reciprocal_basis(va,vb,vc)
    Bstar = np.column_stack([b1,b2,b3])  # 3x3
    R = R_ZXZ(*angles)

    hkls = []
    for h in range(-Hmax, Hmax+1):
        for k in range(-Hmax, Hmax+1):
            for l in range(-Hmax, Hmax+1):
                if h==k==l==0: continue
                if not allow_hkl(bravais, h,k,l):
                    continue
                hkls.append((h,k,l))
    hkls = np.array(hkls, dtype=int)

    # 3D vectors for those reflections
    G = (Bstar @ hkls.T).T  # N x 3
    # Orientation
    V = (R @ G.T).T
    # Unit direction vectors
    norms = np.linalg.norm(V, axis=1)
    U = V / norms[:,None]

    pts, mask = project_to_detector(U, L=L)

    # Limit number of points for speed
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0]-1, max_points, dtype=int)
        pts = pts[idx]
    return pts


def match_score(pred: np.ndarray, obs: np.ndarray, tol: float = 1.5) -> Dict[str, float]:
    if len(pred) == 0:
        return dict(rmse=np.inf, inliers=0, coverage=0.0, score=np.inf)
    tree = cKDTree(pred)
    d, idx = tree.query(obs, k=1)
    inlier_mask = d <= tol
    nin = np.count_nonzero(inlier_mask)
    if nin == 0:
        return dict(rmse=np.inf, inliers=0, coverage=0.0, score=np.inf)
    rmse = float(np.sqrt(np.mean(d[inlier_mask]**2)))
    coverage = nin / obs.shape[0]
    # Robust composite score: lower is better
    # Penalize lack of coverage strongly; prefer low RMSE.
    score = rmse / max(coverage, 1e-6)
    return dict(rmse=rmse, inliers=int(nin), coverage=float(coverage), score=float(score))

# Objective for optimizer

def objective_factory(lattice: LatticeParam, Hmax: int, Ldet: float, obs: np.ndarray, tol: float):
    def obj(p: np.ndarray) -> float:
        # Lattice-specific mapping of parameters and extraction of angles
        if lattice.name == 'triclinic_P':
            cell = lattice.build_cell(p[:6])
            angs = p[6:9]
        elif lattice.name in ['orthorhombic_P','orthorhombic_C','orthorhombic_I','orthorhombic_F']:
            cell = lattice.build_cell(p[:3])
            angs = p[3:6]
        elif lattice.name in ['tetragonal_P','tetragonal_I']:
            cell = lattice.build_cell(p[:2])
            angs = p[2:5]
        elif lattice.name in ['cubic_P','cubic_I','cubic_F']:
            cell = lattice.build_cell(p[:1])
            angs = p[1:4]
        elif lattice.name in ['monoclinic_P','monoclinic_C']:
            cell = lattice.build_cell(p[:4])
            angs = p[4:7]
        elif lattice.name in ['rhombohedral_R']:
            cell = lattice.build_cell(p[:2])
            angs = p[2:5]
        elif lattice.name in ['hexagonal_P']:
            cell = lattice.build_cell(p[:2])
            angs = p[2:5]
        else:
            raise RuntimeError('Unknown lattice config')
        pred = make_predicted_points(lattice.name, cell, angs, Hmax=Hmax, L=Ldet)
        metrics = match_score(pred, obs, tol)
        return metrics['score']
    return obj

# -------------------- Main search loop --------------------

def fit_all_lattices(obs: np.ndarray, Ldet: float = 15.0, Hmax: int = 6, tol: float = 1.5,
                     N_STARTS: int = 12, maxiter: int = 60, seed: int = 123) -> List[Dict]:
    rng = np.random.default_rng(seed)
    results = []
    for lattice in LATTICES:
        lo, hi = lattice.bounds()
        # Differential evolution bounds: append angle bounds after cell params
        # (bounds are already inclusive of angles per lattice.bounds())
        bounds = list(zip(lo, hi))
        best = dict(score=np.inf)
        objective = objective_factory(lattice, Hmax, Ldet, obs, tol)
        for s in range(N_STARTS):
            # Randomize initial population center by jittering bounds
            de = differential_evolution(objective, bounds=bounds, maxiter=maxiter, popsize=12, tol=1e-6,
                                        mutation=(0.5, 1.0), recombination=0.7, polish=False,
                                        seed=rng.integers(1, 10_000_000))
            score = float(de.fun)
            # Recompute detailed metrics at optimum
            p = de.x
            # unpack like in objective
            if lattice.name == 'triclinic_P':
                cell = lattice.build_cell(p[:6]); angs = p[6:9]
            elif lattice.name in ['orthorhombic_P','orthorhombic_C','orthorhombic_I','orthorhombic_F']:
                cell = lattice.build_cell(p[:3]); angs = p[3:6]
            elif lattice.name in ['tetragonal_P','tetragonal_I']:
                cell = lattice.build_cell(p[:2]); angs = p[2:5]
            elif lattice.name in ['cubic_P','cubic_I','cubic_F']:
                cell = lattice.build_cell(p[:1]); angs = p[1:4]
            elif lattice.name in ['monoclinic_P','monoclinic_C']:
                cell = lattice.build_cell(p[:4]); angs = p[4:7]
            elif lattice.name in ['rhombohedral_R']:
                cell = lattice.build_cell(p[:2]); angs = p[2:5]
            elif lattice.name in ['hexagonal_P']:
                cell = lattice.build_cell(p[:2]); angs = p[2:5]
            pred = make_predicted_points(lattice.name, cell, angs, Hmax=Hmax, L=Ldet)
            metrics = match_score(pred, obs, tol)
            cand = dict(lattice=lattice.name, params=p, cell=cell, angles=angs,
                        rmse=metrics['rmse'], coverage=metrics['coverage'],
                        inliers=metrics['inliers'], score=metrics['score'])
            if metrics['score'] < best.get('score', np.inf):
                best = cand
        results.append(best)
    # Rank by score (lower is better)
    results.sort(key=lambda d: d['score'])
    return results


def main():
    # Normalization: re-center spots about origin (optional, but helps robustness)
    obs = spots.copy()
    # Run the fits
    results = fit_all_lattices(obs, Ldet=L, Hmax=6, tol=1.8, N_STARTS=8, maxiter=40, seed=42)

    # Print summary
    print("\n=== Ranked Bravais lattice fits ===")
    print(f"(Lower score is better; score = RMSE / coverage; tol used = 1.8 mm; L = {L} mm)\n")
    print(f"{'Rank':>4}  {'Lattice':<18} {'Score':>10}  {'RMSE[mm]':>9}  {'Cover%':>8}  {'Inliers':>7}")
    for i, r in enumerate(results, 1):
        cover_pct = 100.0 * r['coverage']
        print(f"{i:>4}  {r['lattice']:<18} {r['score']:>10.3f}  {r['rmse']:>9.3f}  {cover_pct:>7.1f}  {r['inliers']:>7}")

    # Quick visual for top-3
    top = results[:3]
    fig, axs = plt.subplots(1, len(top), figsize=(5*len(top), 5), constrained_layout=True)
    if len(top) == 1: axs = [axs]
    for ax, r in zip(axs, top):
        ax.scatter(spots[:,0], spots[:,1], s=12, label='obs')
        pred = make_predicted_points(r['lattice'], r['cell'], r['angles'], Hmax=6, L=L)
        ax.scatter(pred[:,0], pred[:,1], s=6, alpha=0.5, label='pred')
        ax.set_aspect('equal', 'box')
        ax.set_title(f"{r['lattice']}\nscore={r['score']:.2f}, cov={r['coverage']*100:.1f}%")
        ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
        ax.legend(loc='upper right', fontsize=8)
    plt.savefig('fit_summary.png', dpi=160)
    print("\nSaved comparison plot to fit_summary.png")


if __name__ == '__main__':
    main()
