import itertools
import os
from dp_utils.functions import scan_dir_for_ligand_clashes
from dp_utils.helix_assembly.generate_and_score_all_paths import (
    build_permutations_df,
    generate_scored_path_csv,
)
from dp_utils.loop_generation.fix_oxygen_coordinates import fix_all_pdb_oxygen
from dp_utils.loop_generation.fix_rf_diffusion_pdbs import (
    fix_rf_diffusion_pdbs_in_directory,
)
from dp_utils.permutation_data import read_flip_permutations_and_rearrangements
from dp_utils.run_pipeline import fix_pipeline_backbone_oxygens

# from dp_utils.sequence_prediction.ligandmpnn import make_ligand_mpnn_script
from isambard.specifications.deltaprot_helper import Deltahedron
import numpy as np
import pandas as pd
from sklearn import pipeline

from boltz2_utils import make_boltz2_cofold_script, parse_boltz2_dir
from ligandmpnn_utils import select_liganmpnn_sequences, make_ligand_mpnn_script
from pipeline_utils import (
    build_heme_assemblies,
    build_heme_assembly,
    generate_incomplete_paths_csv,
    generate_rfdiffusionaa_inference_lines,
    pipeline_data,
    choose_best_path_per_orientation,
)
from align_all_structures import generate_design_df,align_all_structures


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


# generate_incomplete_paths_csv()

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
# build_heme_assemblies(paths_df, combinations)


# generate_rfdiffusionaa_inference_lines(
#     pipeline_data["directories"]["assembly_output_dir"],
#     os.path.join(
#         pipeline_data["directories"]["rf_diffusion_outputs"],
#         "run_pipeline_diffusion.sh",
#     ),
# )
print("Now run the diffusion script to generate the assemblies.")

# scan_dir_for_ligand_clashes(
#     pipeline_data["directories"]["assembly_output_dir"],
#     num_cpu=18,
# )


# fix_pipeline_backbone_oxygens(
#     source_dir=os.path.join(
#         pipeline_data["directories"]["rf_diffusion_outputs"], "outputs"
#     ),
#     destination_dir=pipeline_data["directories"]["seq_pred_inputs"],
#     cpu_count=18,
# )

# scan_dir_for_ligand_clashes(
#     directory=pipeline_data["directories"]["seq_pred_inputs"],
#     num_cpu=18,
# )


# seq_design_script = make_ligand_mpnn_script(
#     input_dir=pipeline_data["directories"]["seq_pred_inputs"],
#     output_dir=pipeline_data["directories"]["seq_pred_outputs"],
#     run_py="run.py",
#     seed=111,
#     num_seq_per_target=80,
#     batch_size=80,
#     number_of_batches=1,
#     temperature=0.05,
#     checkpoint="./model_params/ligandmpnn_v_32_005_25.pt",
#     gpu_devices="0",  # restrict to first GPU
# )
# print(
#     f"Now run the ligandmpnn script to generate the assemblies at {seq_design_script}"
# )

# Oversample and select up to 1 sequence per design that meets certain criteria
# select_liganmpnn_sequences(
#     ligandmpnn_root_dir=os.path.join(
#         pipeline_data["directories"]["seq_pred_outputs"], "outputs"
#     ),
#     selected_fasta_path=os.path.join(
#         pipeline_data["directories"]["seq_pred_outputs"], "selected_sequences.fasta"
#     ),
#     selected_packed_structures_dir=os.path.join(
#         pipeline_data["directories"]["seq_pred_outputs"], "selected_sequences_packed"
#     ),
# )

# # Cofold with HEM
# make_boltz2_cofold_script(
#         input_fasta=os.path.join(
#         pipeline_data["directories"]["seq_pred_outputs"], "selected_sequences.fasta"
#     ),
#     fold_outputs_base_dir = pipeline_data["directories"]["str_pred_outputs"],
#     fold_with_ligand="HEM",
# )

# # Just fold the sequence
# make_boltz2_cofold_script(
#         input_fasta=os.path.join(
#         pipeline_data["directories"]["seq_pred_outputs"], "selected_sequences.fasta"
#     ),
#     fold_outputs_base_dir = pipeline_data["directories"]["str_pred_outputs"],
# )




design_df = generate_design_df(assemblies_dir = pipeline_data["directories"]["assembly_output_dir"],
                     diffusion_dir = pipeline_data["directories"]["rf_diffusion_outputs"],
                     fold_HEM_dir_root= os.path.join(pipeline_data["directories"]["str_pred_outputs"],"outputs_HEM","boltz_results_inputs_HEM","predictions"),
                     fold_no_lig_dir_root = os.path.join(pipeline_data["directories"]["str_pred_outputs"],"outputs_no_lig","boltz_results_inputs_no_lig","predictions"),
                     save_dir = pipeline_data["directories"]["evaluation"],)

align_df = align_all_structures(design_df,output_dir=pipeline_data["directories"]["evaluation"])
print(align_df)
df_HEM = parse_boltz2_dir(
    main_dir=os.path.join(pipeline_data["directories"]["str_pred_outputs"],"outputs_HEM","boltz_results_inputs_HEM"),
    expect_ligand=True
)
df_no_lig = parse_boltz2_dir(
    main_dir=os.path.join(pipeline_data["directories"]["str_pred_outputs"],"outputs_no_lig","boltz_results_inputs_no_lig"),
    expect_ligand=False
)

# merge these two on orientation_code and sequence_name
df = pd.merge(df_HEM, df_no_lig, how="outer", suffixes=("","_duplicate"))
# print how many _duplicate columns
print(df.columns.str.contains("_duplicate"))


print(df.columns)
print(df.sort_values("confidence_score", ascending=False).head())
# plot with seaborn
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 6))
sns.violinplot(x="orientation_code", y="confidence_score", data=df,cut=0, linewidth=0.6, bw_adjust=0.3)
# save fig as png and svg at /home/tadas/code/deltaprot_heme_binder_design/paths
plt.savefig(os.path.join(pipeline_data["directories"]["str_pred_outputs"],"boltz2_folds_test.png"))
