import itertools
import os
from dpFinder.find_deltaprots import find_deltaprots
from dp_utils.functions import scan_dir_for_ligand_clashes,parse_sequence_prediction_and_plot_propensities
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


from boltz2_utils import make_boltz2_cofold_script, parse_boltz2_dir
from ligandmpnn_utils import select_liganmpnn_sequences, make_ligand_mpnn_script
from pipeline_utils import (
    build_heme_assemblies,
    build_heme_assembly,
    generate_incomplete_paths_csv,
    generate_rfdiffusionaa_inference_lines,
    generate_all_data_df,
    pipeline_data,
    choose_best_path_per_orientation,
)
from align_all_structures import generate_design_df,align_all_structures


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
print(f"Yaw angles: {yaw_angles}")
print(f"Radius distances: {radius_distances}")
print(f"Deltahedron sizes: {deltahedron_sizes}")
print(f"Residues per helix: {residues_per_helix}")
print(f"Linker lengths: {linker_lengths}")

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
#     diffuse_termini=pipeline_data["params"]["diffuse_termini"],
# )
# print("Now run the diffusion script to generate the assemblies.")


# fix_pipeline_backbone_oxygens(
#     source_dir=os.path.join(
#         pipeline_data["directories"]["rf_diffusion_outputs"], "outputs"
#     ),
#     destination_dir=pipeline_data["directories"]["seq_pred_inputs"],
#     cpu_count=18,
# )

scan_dir_for_ligand_clashes(
    directory=pipeline_data["directories"]["seq_pred_inputs"],
    num_cpu=18,
    output_dir=pipeline_data["directories"]["evaluation"],
)


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
# parse_sequence_prediction_and_plot_propensities(
#     ligandmpnn_output_dir=os.path.join(
#         pipeline_data["directories"]["seq_pred_outputs"], "outputs"
#     ),
#     output_dir=os.path.join(
#         pipeline_data["directories"]["evaluation"]
#     )
# )
# # Oversample and select up to 1 sequence per design that meets certain criteria
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




# design_df = generate_design_df(assemblies_dir = pipeline_data["directories"]["assembly_output_dir"],
#                      diffusion_dir = pipeline_data["directories"]["rf_diffusion_outputs"],
#                      fold_HEM_dir_root= os.path.join(pipeline_data["directories"]["str_pred_outputs"],"outputs_HEM","boltz_results_inputs_HEM","predictions"),
#                      fold_no_lig_dir_root = os.path.join(pipeline_data["directories"]["str_pred_outputs"],"outputs_no_lig","boltz_results_inputs_no_lig","predictions"),
#                      save_dir = pipeline_data["directories"]["evaluation"],)

# align_df = align_all_structures(design_df,output_dir=pipeline_data["directories"]["evaluation"])


# df_HEM = parse_boltz2_dir(
#     main_dir=os.path.join(pipeline_data["directories"]["str_pred_outputs"],"outputs_HEM","boltz_results_inputs_HEM"),
#     expect_ligand=True
# )
# df_no_lig = parse_boltz2_dir(
#     main_dir=os.path.join(pipeline_data["directories"]["str_pred_outputs"],"outputs_no_lig","boltz_results_inputs_no_lig"),
#     expect_ligand=False
# )
# df = pd.merge(df_HEM, df_no_lig, how="outer", suffixes=("","_duplicate"))
# # print how many _duplicate columns
# print(df.columns.str.contains("_duplicate"))
# # save to csv
# df.to_csv(os.path.join(pipeline_data["directories"]["evaluation"],"boltz2_info.csv"), index=False)




# find_deltaprots(pdb_dir=os.path.join(pipeline_data["directories"]["str_pred_outputs"],"outputs_no_lig","boltz_results_inputs_no_lig"),
#     output_dir=pipeline_data["directories"]["evaluation"],
#     out_filename="dp_finder_results_no_lig",num_cores=20,search_n_subdirs = 2
# )
# find_deltaprots(pdb_dir=os.path.join(pipeline_data["directories"]["str_pred_outputs"],"outputs_HEM","boltz_results_inputs_HEM"),
#     output_dir=pipeline_data["directories"]["evaluation"],
#     out_filename="dp_finder_results_HEM",num_cores=20,search_n_subdirs = 2
# )

# print("Run get_esm_log_likelihood.py to compute PLL scores for the sequences in the selected_sequences.fasta file.")


# print(df.columns)
# print(df.sort_values("confidence_score", ascending=False).head())
# # plot with seaborn
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.figure(figsize=(20, 6))
# sns.violinplot(x="orientation_code", y="confidence_score", data=df,cut=0, linewidth=0.6, bw_adjust=0.3)
# # save fig as png and svg at /home/tadas/code/deltaprot_heme_binder_design/paths
# plt.savefig(os.path.join(pipeline_data["directories"]["str_pred_outputs"],"boltz2_folds_test.png"))


# generate_all_data_df(pipeline_data["directories"]["evaluation"],os.path.join(pipeline_data["directories"]["seq_pred_outputs"],"selected_sequences.fasta"))