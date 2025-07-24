import pandas as pd
from pathlib import Path
import os
import ampal
from pipeline_utils import (
    build_heme_assemblies,
    build_heme_assembly,
    generate_incomplete_paths_csv,
    generate_rfdiffusionaa_inference_lines,
    generate_all_data_df,
    pipeline_data,
    choose_best_path_per_orientation,
)

from dp_utils.functions import copy_files_to_local_directory
from align_all_structures import align_all_structures
from pipeline_utils import useful_cols
from dp_utils.functions import copy_files_to_local_directory
from align_all_structures import align_all_structures
from ligandmpnn_utils import heme_covalent_cys
from pipeline_utils import useful_cols
import ampal

from align_all_structures import align_structures, save_pymol_session

def normalize_metrics(df, sort_criteria):
    """
    Returns a new DataFrame with columns `xxx_norm` for each metric in `sort_criteria`.
    The value in sort_criteria is a numeric modifier:
      - Its sign indicates the desired direction: negative means lower is better,
        positive means higher is better.
      - Its absolute value indicates the relative importance of that metric.
    Normalization is based on the full df, not a subset.
    """
    result = df.copy()
    for col, modifier in sort_criteria.items():
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            norm_vals = (df[col] - col_min) / (col_max - col_min)
        else:
            print(f"Warning: All values are the same for {col}")
            norm_vals = 0.5  # fallback value
        
        # If modifier is negative, lower values are better. Invert the normalized value.
        if modifier < 0:
            norm_vals = 1 - norm_vals
        
        result[f"{col}_norm"] = norm_vals
    return result


def generate_fasta_string_from_df(df,name_col,seq_col):
    fasta_string = ""
    for index, row in df.iterrows():
        fasta_string += f">{row[name_col]}\n{row[seq_col]}\n"
    return fasta_string



def rank_heme_binder_designs(df, use_hard_filters=True):

    df["dp_finder_total_cost_ignore_1_missing"] = df["dp_finder_total_cost"]-(15.0)/ df["orientation_code"].apply(lambda x: int(x[1]))
    sort_criteria = {
    
    # good sequence (total weight: 1)
        # Cost effectiveness
        'atp_cost_per_aa': -0.1,  
        'sequence_length': -0.1,        
        # Codon variety  
        'dna_complexity_per_aa': 0.2,  
        # esm metric
        'pll_per_aa': 0.2,
        # NetSolP
        'predicted_usability': 0.4,    

    # good apo-protein (total weight: 1):
        'boltz2_no_lig_complex_plddt': 0.8,        
        'boltz2_no_lig_ptm': 0.2,      

    # good holo-protein (total weight: 2):
        # boltz2 holo-protein health 
        'boltz2_complex_plddt': 0.8,        
        'boltz2_ptm': 0.2,      
        # boltz2 ligand binding 
        'boltz2_complex_ipde': -0.25,
        'boltz2_ligand_iptm': 0.25,
        'boltz2_affinity_pred_value2_kcal_mol': -0.25,
        'boltz2_affinity_probability_binary2': 0.25,
        'gnina_affinity':-0.25,

    # design integrity (total weight: 1)
        'dp_finder_total_cost_ignore_1_missing': -0.25,
        'rmsd100': -0.25,
        'rmsd_heme': -0.25,
        'rmsd_fe': -0.25,

        # # De Stress              
        # 'aggrescan3d_avg_value': -1,
        # 'hydrophobic_fitness': -0.5, 
        # 'packing_density': 0.5,
        # 'rosetta_total_per_aa': -0.5,
    }
    df["covalent_heme_attachements"] = df["fold_HEM_pdb_path"].apply(
        lambda x: heme_covalent_cys(x, threshold=3.5)
    )
    if use_hard_filters:
        # Hard Filters
        df = df[
            (df["sequence_molar_extinction_280"] > 1000) &
            (df["rmsd100"] < 5) &
            (df["boltz2_complex_plddt"] > 0.7) &
            (df["dp_finder_best_fit_orientation_expected"] == True) &
            (df["dp_finder_total_unassigned_axes"] == 1) &
            (df["covalent_heme_attachements"] == 0) &
            (df["heme_binding_residues_present"] == True)
        ]

    filtered_df = normalize_metrics(df, sort_criteria)

    filtered_df = filtered_df[
        (filtered_df["covalent_heme_attachements"] == 0)]

    # Soft Filters
    weights = {f"{col}_norm": abs(modifier) for col, modifier in sort_criteria.items()}
    total_weight = sum(weights.values())
    
    # Calculate the weighted average for each row.
    filtered_df["combined_score"] = filtered_df.apply(
        lambda row: sum(row[col] * weight for col, weight in weights.items()) / total_weight,
        axis=1
    )



    # Sort descending so that higher combined scores come first.
    filtered_df.sort_values("combined_score", ascending=False, inplace=True)

    return filtered_df

def save_and_visualize_top_heme_binders(df, selected_deltaprots_dir,label):
    selected_deltaprots_df = df[:30] #filtered_df.drop_duplicates(keep="first",subset="orientation_code")
    
    copy_files_to_local_directory(selected_deltaprots_df["fold_HEM_pdb_path"].tolist(),selected_deltaprots_dir)

    # save df as csv
    # selected_deltaprots_df[useful_cols].to_csv(os.path.join(selected_deltaprots_dir,f"{label}_selected_deltaprots.csv"),index=False)

    pd.to_pickle(selected_deltaprots_df,os.path.join(selected_deltaprots_dir,"selected_deltaprots.pkl"))

    fasta_string =generate_fasta_string_from_df(selected_deltaprots_df,"sequence_name","sequence")
    with open(os.path.join(selected_deltaprots_dir, f"{label}_selected_deltaprots.fasta"),"w") as f:
        f.write(fasta_string)

    # 6. Align structures & save PyMOL sessions
    # align_all_structures will create aligned_pse/ and rmsd_df.csv under selected_deltaprots_dir

    for _, row in selected_deltaprots_df.iterrows():
        session_dir = Path(selected_deltaprots_dir) / "aligned_pse"
        # Align structures and save PyMOL session
        metrics, assemblies = align_structures(
            Path(row["assembly_pdb_path"]),
            Path(row["diffusion_pdb_path"]),
            Path(row["fold_HEM_pdb_path"]),
            Path(row["fold_no_lig_pdb_path"]),
            row["sequence_name"],
        )

        # Save session if within thresholds
        session_dir.mkdir(parents=True, exist_ok=True)
        pse_path = session_dir / f"{row['sequence_name']}.pse"
        pdbs = {name: asm.make_pdb() for name, asm in assemblies.items()}
        print(pdbs.keys())
        save_pymol_session(pdbs, str(pse_path), by_string=True)

    



def main():
    # generate_all_data_df(pipeline_data["directories"]["evaluation"],os.path.join(pipeline_data["directories"]["seq_pred_outputs"],"selected_sequences.fasta"))
    all_info = pd.read_pickle("/home/tadas/code/deltaprot_heme_binder_design/outputs/6_evaluation/all_info.pkl")
    ranked_df = rank_heme_binder_designs(all_info,use_hard_filters=True)
    save_and_visualize_top_heme_binders(ranked_df, "/home/tadas/code/deltaprot_heme_binder_design/outputs/6_evaluation/overall_best_top_30","deltaprot_heme_binder_design",)

if __name__ == "__main__":
    main()


