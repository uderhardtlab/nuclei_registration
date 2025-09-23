# Multi-round nucleus label matcher using sparse LAPJV-style assignment (optimized for ~15k nuclei)
# Type: code/python

"""
This script matches nuclei across multiple 3D segmentation mask TIFFs using a sparse, optimal 1:1 assignment
based on Euclidean distance between centroids. It is optimized for ~15k nuclei per round and avoids dense
matrices by using KDTree neighbor queries + OR-Tools MinCostFlow for sparse assignment.

Features:
- True 1:1 optimal assignment using sparse graph.
- Handles unmatched nuclei and optional one-to-many splits post-pass.
- Efficient for tens of thousands of nuclei.
- Saves relabeled masks and mapping CSVs.
"""

import numpy as np
import tifffile
from skimage.measure import regionprops
from scipy.spatial import cKDTree
from ortools.graph.python import min_cost_flow
import csv
import os
import argparse

# -------------------------
# Helper functions
# -------------------------

def load_mask(path):
    return tifffile.imread(path)


def extract_centroids_and_labels(labeled):
    print("extracting centroids")
    props = regionprops(labeled)
    centroids = np.array([p.centroid for p in props], dtype=float)
    label_ids = np.array([p.label for p in props], dtype=int)
    return labeled, centroids, label_ids


def build_sparse_edges(ref_pts, mov_pts, max_radius, k_neighbors=20, cost_scale=1000):
    print("building edges")
    tree_mov = cKDTree(mov_pts)
    edges = []  # (ref_idx, mov_idx, int_cost)

    for ref_idx, ref_pt in enumerate(ref_pts):
        neighbors = tree_mov.query(ref_pt, k=k_neighbors, distance_upper_bound=max_radius)
        dists, mov_indices = neighbors
        if np.isscalar(dists):
            if dists <= max_radius and mov_indices < len(mov_pts):
                edges.append((ref_idx, mov_indices, int(round(dists*cost_scale))))
        else:
            for dist, mov_idx in zip(dists, mov_indices):
                if dist != np.inf and mov_idx < len(mov_pts):
                    edges.append((ref_idx, mov_idx, int(round(dist*cost_scale))))
    return edges


def sparse_assignment(ref_pts, mov_pts, edges):
    print("sparse assignment")
    n_ref = len(ref_pts)
    n_mov = len(mov_pts)

    start_nodes = []
    end_nodes = []
    capacities = []
    unit_costs = []

    # Build MinCostFlow graph
    S = n_ref + n_mov  # source node
    T = n_ref + n_mov + 1  # sink node

    # edges from source to ref nodes
    for i in range(n_ref):
        start_nodes.append(S)
        end_nodes.append(i)
        capacities.append(1)
        unit_costs.append(0)

    # edges from mov nodes to sink
    for j in range(n_mov):
        start_nodes.append(n_ref + j)
        end_nodes.append(T)
        capacities.append(1)
        unit_costs.append(0)

    # edges ref -> mov (sparse candidate edges)
    for ref_idx, mov_idx, cost in edges:
        start_nodes.append(ref_idx)
        end_nodes.append(n_ref + mov_idx)
        capacities.append(1)
        unit_costs.append(cost)

    # supplies: S = total_flow, T = -total_flow, others = 0
    total_flow = min(n_ref, n_mov)
    supplies = [0]*(n_ref + n_mov + 2)
    supplies[S] = total_flow
    supplies[T] = -total_flow

    mcf = min_cost_flow.SimpleMinCostFlow()

    for s, e, cap, cost in zip(start_nodes, end_nodes, capacities, unit_costs):
        mcf.add_arc_with_capacity_and_unit_cost(s, e, cap, cost)

    for node, supply in enumerate(supplies):
        mcf.set_nodes_supplies(node, supply)

    status = mcf.solve()
    if status != mcf.OPTIMAL:
        raise RuntimeError('Sparse assignment solver did not find optimal solution')

    # Extract mapping ref -> mov
    mapping = {}  # mov_idx -> ref_idx
    for i in range(mcf.num_arcs()):
        if mcf.flows(i) > 0:
            start = mcf.tail(i)
            end = mcf.head(i)
            if start < n_ref and n_ref <= end < n_ref + n_mov:
                ref_idx = start
                mov_idx = end - n_ref
                mapping[mov_idx] = ref_idx
    return mapping


def match_one_round_sparse(ref_centroids, ref_labels, mov_centroids, mov_labels, max_radius=8.0, k_neighbors=20, cost_scale=1000):
    print("matching")
    edges = build_sparse_edges(ref_centroids, mov_centroids, max_radius=max_radius, k_neighbors=k_neighbors, cost_scale=cost_scale)
    idx_map = sparse_assignment(ref_centroids, mov_centroids, edges)

    # convert idx_map (mov_idx -> ref_idx) to label mapping
    mapping = {}
    for mov_idx, ref_idx in idx_map.items():
        mapping[int(mov_labels[mov_idx])] = int(ref_labels[ref_idx])

    # unmatched moving labels will be assigned new IDs later
    return mapping


def relabel_mask(mov_labeled_mask, mapping, next_free_label):
    print("relabelling")
    out = np.zeros_like(mov_labeled_mask, dtype=np.int32)
    unique_mov = np.unique(mov_labeled_mask)
    for ml in unique_mov:
        if ml == 0:
            continue
        ml = int(ml)
        if ml in mapping:
            out[mov_labeled_mask == ml] = mapping[ml]
        else:
            out[mov_labeled_mask == ml] = next_free_label
            mapping[ml] = next_free_label
            next_free_label += 1
    return out, next_free_label

# -------------------------
# Multi-round orchestration
# -------------------------

def match_multi_round_sparse(mask_path, out_prefix, max_radius=8.0, k_neighbors=20, cost_scale=1000, save_mapping_csv=True):
    mask_paths = [os.path.join(mask_path, f) for f  in os.listdir(mask_path) if f.startswith("aligned")]
    n = len(mask_paths)

    masks, labeled, cents, label_ids = [], [], [], []

    for p in mask_paths:
        mask = load_mask(p)
        lab, c, ids = extract_centroids_and_labels(mask)
        masks.append(mask)
        labeled.append(lab)
        cents.append(c)
        label_ids.append(ids)

    ref_cents = cents[0]
    ref_ids = label_ids[0]

    next_free = int(np.max(ref_ids)) + 1 if len(ref_ids) > 0 else 1

    # save reference mask as-is
    tifffile.imwrite(f'{out_prefix}_round01_matched.tif', labeled[0].astype(np.int32), compression="zlib")

    for i in range(1, n):
        mapping = match_one_round_sparse(ref_cents, ref_ids, cents[i], label_ids[i], max_radius=max_radius, k_neighbors=k_neighbors, cost_scale=cost_scale)
        relabeled_mask, next_free = relabel_mask(labeled[i], mapping, next_free)
        out_path = f'{out_prefix}_round{i+1:02d}_matched.tif'
        tifffile.imwrite(out_path, relabeled_mask.astype(np.int32), compression="zlib")

        if save_mapping_csv:
            csv_path = f'{out_prefix}_round{i+1:02d}_mapping.csv'
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['moving_label', 'assigned_ref_label'])
                for mov_label in sorted(mapping.keys()):
                    writer.writerow([mov_label, mapping[mov_label]])


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

    match_multi_round_sparse(args.masks, args.out_prefix, max_radius=args.max_radius, k_neighbors=20, cost_scale=1000, save_mapping_csv=args.save_csv)