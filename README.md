# Multi-Round 3D Nucleus Label Matching

This pipeline matches nuclei across multiple 3D segmentation mask TIFFs using a **sparse, optimal 1:1 assignment**. It is optimized for large datasets (>10k nuclei per round) and avoids dense matrices by leveraging KDTree neighbor queries combined with OR-Tools Min-Cost Flow.

## Features

- Matches nuclei based on **centroid position, volume, and sphericity**.
- Sparse graph construction for efficiency.
- Handles unmatched nuclei by making them disappear (background) in the moving rounds.
- Saves relabeled masks and mapping CSV files for each round.
- Supports 3D masks directly (Cellpose, Ilastik, or other segmentations).

## Installation

```bash
conda create -n nuc_match python=3.11
conda activate nuc_match
pip install numpy scipy scikit-image tifffile ortools
```

## Usage

```python
match_multi_round_sparse(masks, 
                        out_prefix, 
                        max_distance=max_distance, 
                        k_neighbors=10, 
                        cost_scale=1000,
                        w_pos=w_pos, w_vol=w_vol, w_sph=w_sph, 
                        save_mapping_csv=True, 
                        r_max=r_max, s_max=s_max)

```
- Masks must be 3D TIFFs with integer labels.
- The first mask is used as the **reference round**.
- Subsequent masks will be relabeled to match the reference.

## Feature Weighting

The cost function combines multiple features:

```text
cost = w_pos * distance + w_vol * |log(volume_ref / volume_mov)| + w_sph * |sphericity_ref - sphericity_mov|
```

> These weights balance position, size, and shape differences. You may adjust based on dataset variability.

## Output

For each round:

- **Relabeled mask:** `{out_prefix}_roundXX_matched.tif`
- **Mapping CSV:** `{out_prefix}_roundXX_mapping.csv` (moving_label → assigned_ref_label)
- Prints the number of unmatched nuclei.

## Notes

- Pre-filtering edges based on volume ratio or sphericity difference can improve speed and prevent spurious matches.
- KDTree neighbor search is used to create a sparse candidate graph.
- Max-Flow is optionally computed first to determine the feasible number of matches, followed by Min-Cost Flow for optimal assignment.

## References

- [OR-Tools Min-Cost Flow](https://developers.google.com/optimization/graph/mincostflow)  
- [scikit-image regionprops](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)

