# Multi-round nucleus label matcher (reference = round 1)
# Type: code/python

"""
This script loads an arbitrary number of 3D segmentation mask TIFFs (aligned to the same coordinate system),
computes centroids, then finds one-to-one correspondences from each moving round to the reference round (round 1)
using the Hungarian algorithm. The matched masks are relabeled so that the same nucleus across rounds
uses the reference label. Unmatched nuclei get new unique labels.

Key features:
- Radius-based candidate filtering (fast with KDTree) to keep cost matrices small.
- Uses a large cost for non-candidate pairs so Hungarian works but avoids large memory use when cutoff small.
- Safe handling of missing nuclei, splits, and merges: you can choose whether to allow one-to-many mapping after
  the core 1:1 assignment.
- Saves relabeled masks and a CSV with mapping tables per round.

Usage example:
  python match_labels_multi_round.py --masks aligned_Hoechst-IBEX01_masks.tif aligned_Hoechst-IBEX02_masks.tif aligned_Hoechst-IBEX03_masks.tif \
       --out-prefix IBEX_matched --max-radius 8.0 --allow-split 1

"""

import argparse
import numpy as np
import tifffile
from skimage.measure import regionprops
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import csv
import os

# -------------------------
# Helper functions
# -------------------------

def load_mask(path):
    return tifffile.imread(path)


def extract_centroids_and_labels(labeled):
    """Return labeled_mask, centroids (N x 3), label_ids (len N)
    centroids are returned in (z, y, x) order as floats.
    """
    props = regionprops(labeled)
    centroids = np.array([p.centroid for p in props], dtype=float)
    label_ids = np.array([p.label for p in props], dtype=int)
    return labeled, centroids, label_ids


def build_cost_matrix(ref_pts, mov_pts, max_radius, large_cost=1e6):
    """Build an (R x M) cost matrix where distances > max_radius get large_cost.
    This keeps memory reasonable if max_radius is small relative to sample density.
    """
    R = len(ref_pts)
    M = len(mov_pts)
    if R == 0 or M == 0:
        return np.zeros((R, M), dtype=float)

    tree = cKDTree(mov_pts)
    # For each ref point find moving points within radius
    candidates = tree.query_ball_point(ref_pts, r=max_radius)

    cost = np.full((R, M), large_cost, dtype=float)
    for i, cand in enumerate(candidates):
        if len(cand) > 0:
            # compute true distances for candidates
            dists = np.linalg.norm(mov_pts[cand] - ref_pts[i], axis=1)
            cost[i, cand] = dists
    return cost


def match_one_round(ref_centroids, ref_labels, mov_centroids, mov_labels, max_radius=10.0, allow_split=False):
    """Match moving round to reference round. Returns mapping mov_label -> ref_label (or new label).
    allow_split: if True, after 1:1 assignment attempt to map remaining moving objects to nearest assigned ref (one-to-many).
    """
    mapping = {}  # mov_label -> ref_label
    if len(ref_centroids) == 0:
        # everything gets new labels starting after 0
        for ml in mov_labels:
            mapping[int(ml)] = None
        return mapping

    cost = build_cost_matrix(ref_centroids, mov_centroids, max_radius=max_radius)

    # Hungarian requires a square or rectangular matrix; large_cost prevents far matches
    row_ind, col_ind = linear_sum_assignment(cost)

    # Accept only assignments within max_radius (cost < large_cost/2)
    large_cost = 1e6
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < large_cost/2:
            mapping[int(mov_labels[c])] = int(ref_labels[r])

    # Unmatched moving labels
    unmatched_mov = [int(l) for l in mov_labels if int(l) not in mapping]

    if allow_split and len(unmatched_mov) > 0:
        # Map them to nearest reference (even if that ref already has a match) if within max_radius
        tree_ref = cKDTree(ref_centroids)
        dists, idxs = tree_ref.query(mov_centroids[[np.where(mov_labels==m)[0][0] for m in unmatched_mov]], k=1)
        for m_label, d, idx in zip(unmatched_mov, dists, idxs):
            if d <= max_radius:
                mapping[int(m_label)] = int(ref_labels[idx])

    # Remaining unmapped moving labels will be assigned new ids by caller
    return mapping


def relabel_mask(mov_labeled_mask, mapping, next_free_label):
    """Return a relabeled mask where moving labels are replaced by mapped ref labels or new labels starting at next_free_label.
    Also return updated next_free_label.
    mapping: mov_label -> ref_label or None
    """
    out = np.zeros_like(mov_labeled_mask, dtype=np.int32)
    unique_mov = np.unique(mov_labeled_mask)
    for ml in unique_mov:
        if ml == 0:
            continue
        ml = int(ml)
        if ml in mapping and mapping[ml] is not None:
            out[mov_labeled_mask == ml] = int(mapping[ml])
        else:
            out[mov_labeled_mask == ml] = next_free_label
            mapping[ml] = next_free_label
            next_free_label += 1
    return out, next_free_label

# -------------------------
# Main multi-round function
# -------------------------

def match_multi_round(mask_path, out_prefix, max_radius=8.0, allow_split=False, save_mapping_csv=True):
    """mask_paths: list of file paths ordered so that index 0 is reference round
    Produces relabeled masks: <out_prefix>_roundXX_matched.tif and a mapping CSV per round if requested.
    """
    mask_paths = [os.path.join(mask_path, f) for f  in os.listdir(mask_path) if f.endswith(".tif")]
    n = len(mask_paths)
    print(f"Loading {n} masks...")

    # Load and extract ref (round 1)
    masks = [None] * n
    labeled = [None] * n
    cents = [None] * n
    label_ids = [None] * n

    for i, p in enumerate(mask_paths):
        masks[i] = load_mask(p)
        lab, c, ids = extract_centroids_and_labels(masks[i])
        labeled[i] = lab
        cents[i] = c
        label_ids[i] = ids
        print(f"Round {i+1}: {len(c)} nuclei")

    ref_cents = cents[0]
    ref_ids = label_ids[0]

    # Determine next free label (max of reference labels + 1)
    next_free = int(np.max(ref_ids)) + 1 if len(ref_ids)>0 else 1

    # Save reference mask as-is (already uses reference labels)
    out_ref_path = f"{out_prefix}_round01_matched.tif"
    tifffile.imwrite(out_ref_path, labeled[0].astype(np.int32))
    print(f"Saved reference matched mask: {out_ref_path}")

    # For each moving round, match to reference
    for i in range(1, n):
        print(f"Matching round {i+1} to reference...")
        map_mov_to_ref = match_one_round(ref_cents, ref_ids, cents[i], label_ids[i], max_radius=max_radius, allow_split=allow_split)

        # Relabel moving mask according to mapping
        relabeled_mask, next_free = relabel_mask(labeled[i], map_mov_to_ref, next_free_label=next_free)

        out_path = f"{out_prefix}_round{(i+1):02d}_matched.tif"
        tifffile.imwrite(out_path, relabeled_mask.astype(np.int32))
        print(f"Saved matched mask: {out_path}")

        # optionally save mapping table
        if save_mapping_csv:
            csv_path = f"{out_prefix}_round{(i+1):02d}_mapping.csv"
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['moving_label', 'assigned_ref_label'])
                for mov_label in sorted(list(map_mov_to_ref.keys())):
                    writer.writerow([mov_label, map_mov_to_ref[mov_label] if map_mov_to_ref[mov_label] is not None else 'NEW'])
            print(f"Saved mapping CSV: {csv_path}")

    print("All rounds processed. Labels are now consistent with reference round.")

# -------------------------
# CLI
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match nucleus labels across multiple 3D mask TIFFs (reference = first mask)')
    parser.add_argument('--masks', required=True, help='Paths to mask dir; first is reference')
    parser.add_argument('--out-prefix', default='matched', help='Prefix for output files')
    parser.add_argument('--max-radius', type=float, default=8.0, help='Maximum centroid distance (voxels) to consider for matching')
    parser.add_argument('--allow-split', type=int, choices=[0,1], default=0, help='If 1, allow one-to-many mapping after core 1:1 Hungarian assignment')
    parser.add_argument('--no-csv', dest='save_csv', action='store_false', help='Do not save mapping CSV files')
    args = parser.parse_args()

    match_multi_round(args.masks, args.out_prefix, max_radius=args.max_radius, allow_split=bool(args.allow_split), save_mapping_csv=args.save_csv)
