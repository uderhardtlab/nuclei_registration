import numpy as np
from skimage.measure import regionprops
from scipy.spatial import cKDTree
from ortools.graph.python import max_flow, min_cost_flow
import tifffile
import os
import porespy as ps


# Feature extraction
def extract_centroids_and_labels(labeled): # TODO, spacing):
    props = ps.metrics.regionprops_3D(labeled) # TODO, spacing=spacing)
    centroids = np.array([p.centroid for p in props], dtype=float)
    label_ids = np.array([p.label for p in props], dtype=int)
    volumes = np.array([p.area for p in props], dtype=float)
    sphericities = np.array([p.sphericity for p in props], dtype=float)
    return centroids, label_ids, volumes, sphericities


# Build sparse edges with additional features
def build_sparse_edges(ref_pts, ref_vol, ref_sph, mov_pts, mov_vol, mov_sph, max_distance, k_neighbors, cost_scale, w_pos, w_vol, w_sph, r_max, s_max):
    tree_mov = cKDTree(mov_pts)
    edges = []  # list of (ref_idx, mov_idx, int_cost)

    for ref_idx, ref_pt in enumerate(ref_pts):
        dists, mov_indices = tree_mov.query(ref_pt, k=k_neighbors, distance_upper_bound=max_distance)
        if np.isscalar(dists):
            dists, mov_indices = np.array([dists]), np.array([mov_indices])
        for dist, mov_idx in zip(dists, mov_indices):
            if dist != np.inf and mov_idx < len(mov_pts):
                vol_ratio = ref_vol[ref_idx] / (mov_vol[mov_idx] + 1e-8)
                sph_diff = abs(ref_sph[ref_idx] - mov_sph[mov_idx])
                if not (1/r_max <= vol_ratio <= r_max):
                    continue  # reject edge
                if sph_diff > s_max:
                    continue  # reject edge
                # compute cost normally
                cost = w_pos*dist + w_vol*abs(np.log(vol_ratio)) + w_sph*sph_diff
                edges.append((ref_idx, mov_idx, int(round(cost*cost_scale))))
    return edges


# Max-flow to compute feasible matching size
def compute_max_flow(edges):
    ref_has_edge = sorted(set(ref for ref,_,_ in edges))
    mov_has_edge = sorted(set(mov for _,mov,_ in edges))
    ref_map = {old:i for i, old in enumerate(ref_has_edge)}
    mov_map = {old:i for i, old in enumerate(mov_has_edge)}

    edges_compact = [(ref_map[r], mov_map[m], c) for r,m,c in edges]
    nR, nM = len(ref_has_edge), len(mov_has_edge)
    S, T = nR + nM, nR + nM + 1

    mf = max_flow.SimpleMaxFlow()
    for r in range(nR):
        mf.add_arc_with_capacity(S, r, 1)
    for r, m, _ in edges_compact:
        mf.add_arc_with_capacity(r, nR + m, 1)
    for m in range(nM):
        mf.add_arc_with_capacity(nR + m, T, 1)

    status = mf.solve(S, T)
    if status != mf.OPTIMAL:
        raise RuntimeError("Max flow solver failed")
    return mf, edges_compact, ref_map, mov_map, nR, nM, S, T


# Min-cost flow for optimal assignment
def sparse_assignment(edges):
    if len(edges) == 0:
        return {}  # nothing to match
    mf_max, edges_compact, ref_map, mov_map, nR, nM, S, T = compute_max_flow(edges)
    max_match = mf_max.optimal_flow()
    if max_match == 0:
        return {}

    mcf = min_cost_flow.SimpleMinCostFlow()
    all_arcs = []

    # S -> ref
    for r in range(nR):
        all_arcs.append(mcf.add_arc_with_capacity_and_unit_cost(S, r, 1, 0))
    # ref -> mov
    for r, m, cost in edges_compact:
        all_arcs.append(mcf.add_arc_with_capacity_and_unit_cost(r, nR + m, 1, cost))
    # mov -> T
    for m in range(nM):
        all_arcs.append(mcf.add_arc_with_capacity_and_unit_cost(nR + m, T, 1, 0))

    supplies = [0]*(nR+nM+2)
    supplies[S] = max_match
    supplies[T] = -max_match
    mcf.set_nodes_supplies(np.arange(len(supplies), dtype=np.int32), supplies)

    status = mcf.solve()
    if status != mcf.OPTIMAL:
        raise RuntimeError("Sparse assignment solver did not find optimal solution")

    solution_flows = mcf.flows(all_arcs)
    ref_keys_sorted = sorted(ref_map.keys())
    mov_keys_sorted = sorted(mov_map.keys())

    mapping = {}
    for arc, flow in zip(all_arcs, solution_flows):
        if flow == 0:
            continue
        tail, head = mcf.tail(arc), mcf.head(arc)
        if tail < nR and nR <= head < nR+nM:
            ref_idx = tail
            mov_idx = head - nR
            mapping[mov_keys_sorted[mov_idx]] = ref_keys_sorted[ref_idx]

    return mapping


# Relabel mask
def relabel_mask(mask, mapping):
    max_label = mask.max()
    lut = np.arange(max_label+1, dtype=np.int32)
    
    all_labels = set(np.unique(mask))
    mapped_labels = set(mapping.keys())
    unmatched_labels = all_labels - mapped_labels - {0}  # exclude background
    print("#Unmatched labels (will be lost):", len(unmatched_labels))
    
    for old_label in np.unique(mask):
        if old_label == 0:
            continue
        if old_label in mapping:
            lut[old_label] = mapping[old_label]
        else:
            lut[old_label] = 0
    return lut[mask]


# Multi-round matching
def match_one_round_sparse(ref_cents, ref_labels, ref_vol, ref_sph, mov_cents, mov_labels, mov_vol, mov_sph, max_distance, k_neighbors, cost_scale, w_pos, w_vol, w_sph, r_max, s_max):
    edges = build_sparse_edges(ref_cents, ref_vol, ref_sph,
                               mov_cents, mov_vol, mov_sph,
                               max_distance=max_distance,
                               k_neighbors=k_neighbors,
                               cost_scale=cost_scale,
                               w_pos=w_pos, w_vol=w_vol, w_sph=w_sph, r_max=r_max, s_max=s_max)
    idx_map = sparse_assignment(edges)
    mapping = {int(mov_labels[mov_idx]): int(ref_labels[ref_idx]) for mov_idx, ref_idx in idx_map.items()}
    return mapping

def match_multi_round_sparse(mask_path, out_prefix, max_distance, k_neighbors, cost_scale, w_pos, w_vol, w_sph, save_mapping_csv, r_max, s_max): # TODO, spacing):
    mask_files = [f for f in os.listdir(mask_path) if f.startswith("aligned")]
    mask_files.sort()
    masks = [tifffile.imread(os.path.join(mask_path, f)) for f in mask_files]

    cents_list, labels_list, vol_list, sph_list = [], [], [], []
    for mask in masks:
        c, l, v, e = extract_centroids_and_labels(mask) # TODO, spacing=spacing)
        cents_list.append(c)
        labels_list.append(l)
        vol_list.append(v)
        sph_list.append(e)

    ref_cents, ref_labels, ref_vol, ref_sph = cents_list[0], labels_list[0], vol_list[0], sph_list[0]

    # Save reference mask
    tifffile.imwrite(f"{out_prefix}_round01_matched.tif", masks[0].astype(np.int32), compression="zlib")

    for i in range(1, len(masks)):
        mapping = match_one_round_sparse(ref_cents, ref_labels, ref_vol, ref_sph,
                                         cents_list[i], labels_list[i], vol_list[i], sph_list[i],
                                         max_distance=max_distance, k_neighbors=k_neighbors,
                                         cost_scale=cost_scale,
                                         w_pos=w_pos, w_vol=w_vol, w_sph=w_sph, r_max=r_max, s_max=s_max)
        relabeled_mask = relabel_mask(masks[i], mapping)
        out_path = f"{out_prefix}_round{i+1:02d}_matched.tif"
        tifffile.imwrite(out_path, relabeled_mask.astype(np.int32), compression="zlib")

        if save_mapping_csv:
            import csv
            csv_path = f"{out_prefix}_round{i+1:02d}_mapping.csv"
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['moving_label', 'assigned_ref_label'])
                for mov_label in sorted(mapping.keys()):
                    writer.writerow([mov_label, mapping[mov_label]])


# CLI / example usage
if __name__ == '__main__':
    masks = "/data/bionets/je30bery/point_set_matching/data"
    out_prefix = "/data/bionets/je30bery/point_set_matching/data/IBEX"
    max_distance = 30
    r_max = 1.5 # factor by which volumes of two matched nuclei can differ
    s_max = 0.3 # maximum absolute difference in sphericity for two matched nuclei
    w_pos = 1.0
    w_vol = 50.0
    w_sph = 100.0
    # TODO spacing = [1, 1, 1]
    match_multi_round_sparse(masks, out_prefix, max_distance=max_distance, k_neighbors=10, cost_scale=1000,
                             w_pos=w_pos, w_vol=w_vol, w_sph=w_sph, save_mapping_csv=True, r_max=r_max, s_max=s_max) # TODO, spacing=spacing)
