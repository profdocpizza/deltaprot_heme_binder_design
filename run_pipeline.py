import itertools
from math import comb
import os
from dp_utils.helix_assembly.generate_and_score_all_paths import (
    build_permutations_df,
    generate_scored_path_csv,
)
from dp_utils.permutation_data import read_flip_permutations_and_rearrangements
from isambard.specifications.deltaprot_helper import Deltahedron
import numpy as np

from pipeline_utils import (
    build_heme_assemblies,
    build_heme_assembly,
    generate_incomplete_paths_csv,
    generate_inference_lines,
    pipeline_data,
    choose_best_path_per_orientation,
)


n_orientations = 30
n_best_paths = 1  # one best for each deltaprot
n_flips = 1  # assume C2symmetry of heme molecule (ignore slight assymetry at the hydrophobic part)
yaw_angles = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
radius_distances = [
    0,
    -2,
    -4,
    -6,
]  # removed helix center radius modified with the value in the list (angstrom)
deltahedron_sizes = [12]
loop_lengths = [8, 9, 10, 11, 12, 13]
residues_per_helix = [8]
n_runs = (
    n_orientations
    * n_best_paths
    * n_flips
    * len(yaw_angles)
    * len(radius_distances)
    * len(deltahedron_sizes)
    * len(loop_lengths)
)
print(f"Number of runs: {n_runs}")


generate_incomplete_paths_csv()

all_paths_df = read_flip_permutations_and_rearrangements(
    csv_input_file_path="/home/tadas/code/deltaprot_heme_binder_design/paths/non_symmetric_paths_edge_linkers_only.csv"
)


all_paths_df = read_flip_permutations_and_rearrangements(
    csv_input_file_path="/home/tadas/code/deltaprot_heme_binder_design/paths/non_symmetric_paths_edge_linkers_only.csv"
)
paths_df = choose_best_path_per_orientation(
    all_paths_df,
    out_dir="/home/tadas/code/deltaprot_heme_binder_design/outputs/1_path_choices",
)


yaw_angles = pipeline_data["params"]["yaw_angles"]
radius_distances = pipeline_data["params"]["radius_distances"]
deltahedron_sizes = pipeline_data["params"]["deltahedron_sizes"]
residues_per_helix = pipeline_data["params"]["residues_per_helix"]
linker_lengths = pipeline_data["params"]["loop_lengths"]

combinations = list(
    itertools.product(
        yaw_angles,
        radius_distances,
        deltahedron_sizes,
        residues_per_helix,
        linker_lengths,
    )
)

print(
    f"N combinations = {len(combinations)} x {len(paths_df)} = {len(combinations)*len(paths_df)}"
)

build_heme_assemblies(paths_df, combinations)

generate_inference_lines(
    pipeline_data["dirs"]["assembly_output_dir"],
    os.path.join(
        pipeline_data["dirs"]["diffusion_output_dir"], "run_pipeline_diffusion.sh"
    ),
)
