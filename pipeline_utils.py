import re
import pandas as pd
from sympy import sequence
import yaml

from isambard.specifications.deltaprot import HelixConformation
import ampal

from dp_utils.helix_assembly.utils import apply_transform_to_assembly
import os

import numpy as np

from isambard.specifications.deltaprot import DeltaProt
from dp_utils.helix_assembly.generate_and_score_all_paths import (
    build_permutations_df,
    generate_scored_path_csv,
)
from dp_utils.pipeline_data import get_pipeline_dp_finder_df
from dp_utils.design_evaluation.model_df_generation import get_complexity_and_atp_cost_columns
from dp_utils.design_evaluation.model_df_utils import calculate_rmsd100
def load_pipeline_config():
    pipeline_config_path = (
        "/home/tadas/code/deltaprot_heme_binder_design/config/config.yaml"
    )
    with open(pipeline_config_path) as f:
        pipeline_data = yaml.load(f, Loader=yaml.FullLoader)
    return pipeline_data


pipeline_data = load_pipeline_config()


def make_pipeline_dirs():
    """
    Create necessary directories for the pipeline.ask user for confirmation
    """
    directories = pipeline_data["directories"]
    for dir_path in directories.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        else:
            print(f"Directory already exists: {dir_path}")


make_pipeline_dirs()

def read_fasta_to_df(fasta_path):

    names = []
    seqs = []
    with open(fasta_path, 'r') as fh:
        current_name = None
        current_seq = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # save the previous record
                if current_name is not None:
                    names.append(current_name)
                    seqs.append(''.join(current_seq))
                current_name = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        # don't forget the last one
        if current_name is not None:
            names.append(current_name)
            seqs.append(''.join(current_seq))

    return pd.DataFrame({
        'sequence_name': names,
        'sequence': seqs
    })

def parse_filename(filename):
    pattern = re.compile(
        r"^"
        r"(?P<orientation_code>.+?)"
        r"_yaw(?P<yaw>-?\d+)"
        r"_rad(?P<radius>-?\d+)"
        r"_delt(?P<deltahedron_size>\d+)"
        r"_helix(?P<residues_per_helix>\d+)"
        r"_link(?P<linker_length>\d+)"
    )
    match = pattern.match(filename)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename!r}")
    
    gd = match.groupdict()
    return {
        'orientation_code':          gd['orientation_code'],
        'yaw':                       float(gd['yaw']),
        'radius':                    float(gd['radius']),
        'deltahedron_size':          float(gd['deltahedron_size']),
        'residues_per_helix':        int(gd['residues_per_helix']),
        'linker_length':             int(gd['linker_length']),
    }


def generate_incomplete_paths_csv():
    path_df = build_permutations_df(exclude_ribs=1)

    linker_cols = [
        c for c in path_df.columns if c.startswith("linker") and c.endswith("_length")
    ]
    path_df["minimal_linker"] = path_df[linker_cols].fillna(1).eq(1).all(axis=1)
    path_df = path_df[path_df["minimal_linker"]]

    optimal_helix_rotation_path = "/home/tadas/code/deltaProteinDesignUtils/data/deltaprot_permutation_data/optimised_helix_rotation_edge_length_11_aa_per_helix_10_angle_resolution_1.csv"
    generate_scored_path_csv(
        "/home/tadas/code/deltaprot_heme_binder_design/paths",
        optimal_helix_rotation_path,
        df=path_df,
    )


def choose_best_path_per_orientation(all_paths_df, out_dir):
    all_paths_df.sort_values(by=["path_score_v3"], inplace=True, ascending=False)
    # drop dupicates of the same orientation_code, keep first
    paths_df = all_paths_df.drop_duplicates(subset=["orientation_code"], keep="first")
    # save df as pkl in save_dir /home/tadas/code/deltaprot_heme_binder_design/outputs/path_choices
    file_path = os.path.join(out_dir, "path_choices_data.pkl")
    paths_df.to_pickle(file_path)
    return paths_df


def find_missing_rib(assembly: DeltaProt):
    """
    Identify which pair of vertices (rib) has no helix attached.
    """
    all_vertices = set(range(len(assembly.deltahedron.vertices)))
    used = {v for conf in assembly.helix_conformations for v in conf.rib_vertices}
    missing = sorted(all_vertices - used)
    # Here choose one missing rib (first two indices)
    return (missing[0], missing[1])


def build_heme_assemblies(paths_df, combinations):
    for _, row in paths_df.iterrows():
        # print(
        #     f"Building heme assembly for path: {row['path']}, deltahedron: {row['deltahedron_name']}"
        # )
        for combo in combinations:
            yaw, radius, deltahedron_size, residues_per_helix, linker_length = combo
            # print(f"Yaw: {yaw}, Radius: {radius}, Size: {size}, Linker Length: {linker}")
            build_heme_assembly(
                row, yaw, radius, deltahedron_size, residues_per_helix, linker_length
            )


def build_heme_assembly(
    row, yaw, radius, deltahedron_size, residues_per_helix, linker_length
):
    save_dir = pipeline_data["directories"]["assembly_output_dir"]

    filename = f"{row['orientation_code'].replace('.', '_')}_yaw{yaw}_rad{radius}_delt{deltahedron_size}_helix{residues_per_helix}_link{linker_length}.pdb"
    file_path = os.path.join(save_dir, filename)
    print(f"Row: {row['orientation_code']}, File: {filename}")
    assembly = DeltaProt(
        [
            HelixConformation(tuple(rib_vertices), rotation, residues_per_helix)
            for rib_vertices, rotation in zip(
                row["path"],
                len(row["path"]) * [0],
            )
        ],
        deltahedron_name=row["deltahedron_name"],
    )
    assembly.optimise_helix_rotations(degree_turn=5)

    def compute_rib_center(assembly: DeltaProt, v1, v2) -> None:
        """
        Compute midpoint of the missing rib on the deltahedron.
        """
        p1 = np.array(assembly.deltahedron.vertices[v1])
        p2 = np.array(assembly.deltahedron.vertices[v2])
        rib_center = (p1 + p2) / 2
        return rib_center

    rib_center_coords = compute_rib_center(assembly, *find_missing_rib(assembly))
    deltahedron_center_coords = np.mean(assembly.deltahedron.vertices, axis=0)
    # print(rib_center_coords, deltahedron_center_coords)

    ligand_assembly = ampal.load_pdb(pipeline_data["ligand"]["file_path"])

    apply_yaw_and_radius(
        ligand_assembly, yaw, radius, translate_first=False
    )  # applied on ligand's local coordinates

    ligand_file_name = f"yaw{yaw}_rad{radius}.pdb"
    ligand_file_path = os.path.join(save_dir, "ligand_only", ligand_file_name)
    os.makedirs(os.path.dirname(ligand_file_path), exist_ok=True)
    with open(ligand_file_path, "w") as f:
        f.write(ligand_assembly.pdb)

    assembly, ligand_assembly,
    rib_coords = [assembly.deltahedron.vertices[i] for i in find_missing_rib(assembly)]
    rib_center_coords = np.mean(rib_coords, axis=0)
    deltahedron_center_coords = np.mean(assembly.deltahedron.vertices, axis=0)

    R = get_rib_align_transform(assembly, rib_vertices=find_missing_rib(assembly))

    # rotate about origin then shift into place:
    apply_transform_to_assembly(
        ligand_assembly,
        rotation_matrix=R,
        translation_vector=rib_center_coords,  # rib_center_coords,
        translate_first=False,
    )
    coords = {atom.res_label: atom._vector for atom in ligand_assembly.get_atoms()}
    distance = np.linalg.norm(coords["FE"] - rib_center_coords)

    # print(f'Fe cords{coords["FE"], rib_center_coords}')
    # print coords of FE atom name
    ligand_assembly.get_atoms()
    # apply_transform_to_assembly(
    #     ligand_assembly,
    #     rotation_matrix=np.identity(3),
    #     translation_vector=-np.zeros(3),
    #     translate_first=True
    # )

    # make a subdir and save the ligand only there

    with open(ligand_file_path.replace(".pdb", "_aligned_to_rib.pdb"), "w") as f:
        f.write(ligand_assembly.pdb)

    assembly.append(ligand_assembly[0])

    with open(file_path, "w") as f:
        f.write(assembly.pdb)

    # assert distance+radius < 1e-6, f"Distance too large {distance-radius}. Fe:{coords['FE']} , deltahedron:{rib_center_coords}. Check {file_path}"

    assembly
    return file_path


def get_z_yaw_translate_transform(yaw_deg: float, radius: float):
    """
    Compute a rotation about the Z-axis by yaw_deg (degrees),
    then a translation of +radius along the GLOBAL Y-axis.
    Returns (R, t) where R is 3×3 and t is length-3.
    """
    φ = np.deg2rad(yaw_deg)
    # Rotation about Z
    R = np.array(
        [
            [np.cos(φ), -np.sin(φ), 0],
            [np.sin(φ), np.cos(φ), 0],
            [0, 0, 1],
        ]
    )
    # Always translate along +Y in world coords
    t = np.array([0, radius, 0])
    return R, t


def apply_yaw_and_radius(assembly, yaw, radius, translate_first=False):
    """
    Rotate `assembly` about Y by `yaw`°, then translate it
    +radius along Y (or do translation first if translate_first=True).
    """
    R, t = get_z_yaw_translate_transform(yaw, radius)
    apply_transform_to_assembly(
        assembly,
        rotation_matrix=R,
        translation_vector=t,
        translate_first=translate_first,
    )


# --- Step 1: Define compute_rib_global_axes ---
def compute_rib_global_axes(assembly, rib_vertices=(8, 11), eps=1e-6):
    """
    Compute global axes and visualize all three frames.
    Returns Xg, Yg, Zg, center, and rotation R.
    """
    verts = np.array(assembly.deltahedron.vertices)
    v1, v2 = verts[rib_vertices[0]], verts[rib_vertices[1]]
    center = verts.mean(axis=0)
    rib_center = 0.5 * (v1 + v2)

    # Global frame
    Yg = rib_center - center
    Yg /= np.linalg.norm(Yg)
    ref = v1 - center
    if abs(np.dot(ref, Yg)) / (np.linalg.norm(ref) + eps) > 0.9:
        ref = v2 - center
    proj = ref - np.dot(ref, Yg) * Yg
    Xg = proj / np.linalg.norm(proj)
    Zg = np.cross(Xg, Yg)
    Zg /= np.linalg.norm(Zg)

    # Compute rotation from identity → global
    local_axes = np.eye(3)
    global_axes = np.column_stack((Xg, Yg, Zg))
    R = rotation_procrustes(local_axes, global_axes)

    # Call the visualization helper
    # visualize_rib_alignment(assembly, rib_vertices, Xg, Yg, Zg, center, R)

    return Xg, Yg, Zg, center, R


def rotation_procrustes(local_axes: np.ndarray, global_axes: np.ndarray) -> np.ndarray:
    """
    Given two 3×3 orthonormal bases (columns are basis vectors),
    compute the rotation R such that R @ local_axes = global_axes
    via the Kabsch / orthogonal Procrustes algorithm.

    Parameters
    ----------
    local_axes : ndarray, shape (3,3)
        Columns are the source basis vectors (orthonormal).
    global_axes : ndarray, shape (3,3)
        Columns are the target basis vectors (orthonormal).

    Returns
    -------
    R : ndarray, shape (3,3)
        Proper rotation matrix with det(R)=+1.
    """
    # 1) covariance
    H = local_axes @ global_axes.T

    # 2) SVD
    U, S, Vt = np.linalg.svd(H)

    # 3) build R and correct for reflection if needed
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    return R


def visualize_rib_alignment(
    assembly, rib_vertices, Xg, Yg, Zg, center, R, axis_scale_factor=1.2
):
    """
    Visualize deltahedron vertices and three sets of axes:
      - Local (identity basis, dotted gray)
      - Global (Xg, Yg, Zg, solid colored)
      - Rotated Local (R @ identity, dashed colored)
    Rib vertices in red, others in blue.
    """
    verts = np.array(assembly.deltahedron.vertices)
    rib_set = set(rib_vertices)
    non_rib_idx = [i for i in range(len(verts)) if i not in rib_set]
    rib_idx = [i for i in range(len(verts)) if i in rib_set]

    fig = go.Figure()
    # Plot vertices
    fig.add_trace(
        go.Scatter3d(
            x=verts[non_rib_idx, 0],
            y=verts[non_rib_idx, 1],
            z=verts[non_rib_idx, 2],
            mode="markers",
            marker=dict(size=5, color="blue"),
            name="Non-Rib Vertices",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=verts[rib_idx, 0],
            y=verts[rib_idx, 1],
            z=verts[rib_idx, 2],
            mode="markers",
            marker=dict(size=7, color="red"),
            name="Rib Vertices",
        )
    )

    axis_scale = np.linalg.norm(verts - center, axis=1).max() * axis_scale_factor

    # Helper to add axes traces
    def _add_axes(basis, prefix, colors, dash):
        for i, ax in enumerate(["X", "Y", "Z"]):
            vec = basis[:, i]
            fig.add_trace(
                go.Scatter3d(
                    x=[center[0], center[0] + vec[0] * axis_scale],
                    y=[center[1], center[1] + vec[1] * axis_scale],
                    z=[center[2], center[2] + vec[2] * axis_scale],
                    mode="lines+markers",
                    marker=dict(size=4),
                    line=dict(width=4, color=colors[ax], dash=dash),
                    name=f"{prefix}-{ax}",
                )
            )

    # Local axes (identity)
    local_axes = np.eye(3)
    colors_local = {"X": "gray", "Y": "lightgray", "Z": "silver"}
    _add_axes(local_axes, "Local", colors_local, dash="dot")

    # Global axes
    global_axes = np.column_stack((Xg, Yg, Zg))
    colors_global = {"X": "green", "Y": "orange", "Z": "purple"}
    _add_axes(global_axes, "Global", colors_global, dash="solid")

    # Rotated local axes
    rotated_axes = R @ local_axes
    colors_rot = {"X": "cyan", "Y": "magenta", "Z": "yellow"}
    _add_axes(rotated_axes, "Rotated", colors_rot, dash="dash")

    fig.update_layout(
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        title="Δ-Hedron & Axes: Local vs Global vs Rotated",
    )
    fig.show()


def rotation_from_axes(local_axes, global_axes):
    """
    Task 2: given two orthonormal bases (3×3 matrices whose columns are
    the basis vectors), find R so that R @ local_axes[:,i] = global_axes[:,i].
    """
    # since local_axes is orthonormal, inv = transpose:
    return global_axes @ local_axes.T


def get_rib_align_transform(assembly, rib_vertices=(8, 11)):
    Xg, Yg, Zg, center, R = compute_rib_global_axes(assembly, rib_vertices=rib_vertices)
    return R.T


def generate_rfdiffusionaa_inference_lines(folder_path, output_script_path):
    seen_lines = set()
    output_lines = []

    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".pdb"):
            continue

        parsed = parse_filename(fname)
        if not parsed:
            print(f"Skipping: {fname} (pattern mismatch)")
            continue

        orientation_code = parsed["orientation_code"]
        yaw = parsed["yaw"]
        radius = parsed["radius"]
        deltahedron_size = parsed["deltahedron_size"]
        residues_per_helix = int(parsed["residues_per_helix"])
        linker_length = int(parsed["linker_length"])

        n_helix = int(orientation_code[1]) - 1
        helices = [chr(ord("A") + i) for i in range(n_helix)]

        # inpaint_seq
        inpaint_seq = ",".join(f"{h}1-{residues_per_helix}" for h in helices)
        inpaint_seq = f"'{inpaint_seq}'"
        # contigs with linker
        # linker_start = residues_per_helix + 1
        # linker_end = linker_start + linker_length - 1
        contig_parts = [
            f"{h}1-{residues_per_helix},{linker_length}-{linker_length}"
            for h in helices[:-1]
        ]
        contig_parts.append(f"{helices[-1]}1-{residues_per_helix}")
        contigs = ",".join(contig_parts)
        contigs = f"'{contigs}'"
        # total length
        total_length = n_helix * residues_per_helix + (n_helix - 1) * linker_length

        pdb_path = os.path.join(folder_path, fname)
        output_prefix = os.path.join(
            pipeline_data["directories"]["rf_diffusion_outputs"],
            "outputs",
            fname.replace(".pdb", ""),
        )

        line = (
            f"PYTHONHASHSEED=1 "
            f"python run_inference.py inference.deterministic=True diffuser.T=20 "
            f"inference.output_prefix={output_prefix} "
            f"inference.input_pdb={pdb_path} "
            f'contigmap.inpaint_seq="[{inpaint_seq}]" '
            f'contigmap.contigs="[{contigs}]" '
            f'contigmap.length="{total_length}-{total_length}" '
            f"inference.ligand=HEM inference.num_designs=1 inference.design_startnum=0"
        )

        if line not in seen_lines:
            seen_lines.add(line)
            output_lines.append(line)

    # assert no duplicates
    assert len(output_lines) == len(seen_lines), "Duplicate lines found!"
    # Write to shell script
    with open(output_script_path, "w") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"Written {len(output_lines)} unique commands to {output_script_path}")


def generate_all_data_df(evaluation_dir,sequences_fasta_path):
    
    # load design_df
    design_df = pd.read_csv(os.path.join(evaluation_dir, "design_df.csv"))
    design_df = design_df.assign(**design_df["sequence_name"].apply(parse_filename).apply(pd.Series))

    # load sequences
    design_df = design_df.merge(
        read_fasta_to_df(sequences_fasta_path),
        on="sequence_name",
        how="left",
        suffixes=("", "_duplicate1"),
    )

    # add sequence features
    design_df["mass"] = design_df["sequence"].apply(ampal.analyse_protein.sequence_molecular_weight)
    design_df["sequence_length"] = design_df["sequence"].apply(len)
    design_df["sequence_molar_extinction_280"] = design_df["sequence"].apply(ampal.analyse_protein.sequence_molar_extinction_280)
    design_df["charge"] = design_df["sequence"].apply(ampal.analyse_protein.sequence_charge)
    design_df["isoelectric_point"] = design_df["sequence"].apply(ampal.analyse_protein.sequence_isoelectric_point)
    get_complexity_and_atp_cost_columns(design_df,sequence_col="sequence")

    # add his tag prefix and recalculate features
    design_df["sequence_w_prefix"] = "MGSSHHHHHHSSGENLYFQSGS" + design_df["sequence"]
    design_df["mass_w_prefix"] = design_df["sequence_w_prefix"].apply(ampal.analyse_protein.sequence_molecular_weight)
    design_df["sequence_length_w_prefix"] = design_df["sequence_w_prefix"].apply(len)
    design_df["sequence_molar_extinction_280_w_prefix"] = design_df["sequence_w_prefix"].apply(ampal.analyse_protein.sequence_molar_extinction_280)
    design_df["charge_w_prefix"] = design_df["sequence_w_prefix"].apply(ampal.analyse_protein.sequence_charge)
    design_df["isoelectric_point_w_prefix"] = design_df["sequence_w_prefix"].apply(ampal.analyse_protein.sequence_isoelectric_point)
    
    # load boltz2_info.csv and merge by sequence_name
    boltz2_df = pd.read_csv(os.path.join(evaluation_dir, "boltz2_info.csv"))
    # add prefix boltz2_ to all columns except sequence_name
    boltz2_df.columns = [
        f"boltz2_{col}" if col != "sequence_name" else col for col in boltz2_df.columns
    ]
    # merge design_df with boltz2_df on sequence_name

    design_df = design_df.merge(
        boltz2_df,
        on="sequence_name",
        how="left",
        suffixes=("", "_duplicate"),
    )
    # load rmsd_df.csv
    design_df = design_df.merge(
        pd.read_csv(os.path.join(evaluation_dir, "rmsd_df.csv")),
        on="sequence_name",
        how="left",
        suffixes=("", "_duplicate2"),
    )


    design_df["rmsd100"] = design_df.apply(
        lambda row: calculate_rmsd100(row["rmsd"], row["sequence_length"]), axis=1
    )
    design_df["rmsd100_no_lig"] = design_df.apply(
        lambda row: calculate_rmsd100(row["rmsd_no_lig"], row["sequence_length"]), axis=1
    )   
    design_df["rmsd100_diffusion"] = design_df.apply(
        lambda row: calculate_rmsd100(row["rmsd_diffusion"], row["sequence_length"]), axis=1
    )


    # load esm log likelihood pll.csv
    esm_log_likelihood_df = pd.read_csv(
        os.path.join(evaluation_dir, "pll.csv")
    ).rename(
        columns={"name": "sequence_name"}
    ).drop(columns=
        ["sequence"]
    )  
    design_df = design_df.merge(
        esm_log_likelihood_df,
        on="sequence_name",
        how="left",
        suffixes=("", "_duplicate22"),
    )
    design_df["pll_per_aa"] = design_df["pll"] / design_df["sequence_length"]

    # load dp_finder_results
    dp_finder_results_HEM_df = get_pipeline_dp_finder_df(os.path.join(evaluation_dir, "dp_finder_results_HEM.pkl"),expect_missing_ribs=1)
    dp_finder_results_no_lig_df = get_pipeline_dp_finder_df(os.path.join(evaluation_dir, "dp_finder_results_no_lig.pkl"),expect_missing_ribs=1)


    # generate sequence_name column for both dfs from filename by removing _model_0.pdb
    dp_finder_results_HEM_df["sequence_name"] = dp_finder_results_HEM_df["dp_finder_filename"].str.replace("_model_0.pdb", "")
    dp_finder_results_no_lig_df["sequence_name"] = dp_finder_results_no_lig_df["dp_finder_filename"].str.replace("_model_0.pdb", "")
    # drop filename column from both dfs
    dp_finder_results_HEM_df.drop(columns=["dp_finder_filename"], inplace=True)
    dp_finder_results_no_lig_df.drop(columns=["dp_finder_filename"], inplace=True)

    # add prefix no_lig_ to all columns of  dp_finder_results_no_lig_df
    dp_finder_results_no_lig_df.columns = [
        f"no_lig_{col}" if col != "sequence_name" else col for col in dp_finder_results_no_lig_df.columns
    ]
    # merge by sequence_name
    design_df = design_df.merge(dp_finder_results_HEM_df, on="sequence_name", how="left", suffixes=("", "_duplicate3"))
    design_df = design_df.merge(dp_finder_results_no_lig_df, on="sequence_name", how="left", suffixes=("", "_duplicate4"))

    # load netsolp.csv
    netsolp_df = pd.read_csv(os.path.join(evaluation_dir, "netsolp.csv")).rename(
        columns={"sid": "sequence_name"}
    )[["sequence_name","predicted_usability"]]
    design_df = design_df.merge(
        netsolp_df,
        on="sequence_name",
        how="left",
        suffixes=("", "_duplicate5"),
    )


    print(design_df)
    print(design_df.columns)

    # save this df to pkl
    design_df.to_pickle(os.path.join(evaluation_dir, "all_info.pkl"))
    return design_df


gene_synthesis_cols = ["Well Position","Name","Sequence"]
sharing_cols = gene_synthesis_cols + [
    "orientation_code",
    "isoelectric_point",
    "charge",
    "mass",
    "sequence_length",
    "mean_plddt",
    "mean_pae",
    "ptm",
    "tm_rmsd100",
    "dp_finder_total_cost",
    "sequence_molar_extinction_280",
    "model_sequence"
]
metrics_cols = [
    "orientation_code",
    "atp_cost_per_aa",
    "atp_cost",
    "dna_complexity_per_aa",
    "pll",
    "pll_per_aa",
    "isoelectric_point",
    "charge",
    # "sequence_charge" redundant to charge
    "mass",
    "sequence_length",
    "mean_plddt",
    "mean_pae",
    "ptm",
    "tm_rmsd",
    "tm_score_assembly",
    "tm_score_design",
    "tm_rmsd100",
    # "n_potential_disulfide_bonds",
    "dp_finder_total_cost",
    "dp_finder_total_scale",

    "predicted_usability",
    "combined_score",
    "path_score_version",
    "residues_per_helix",
    "deltahedron_edge_length",
    "diffuse_termini",
    "avoid_amino_acids",
    "increase_amino_acid_likelihood",
    "energy_minimization",
    "algorithms_sequence_prediction",
    "algorithms_structure_prediction",
    "rf_file_num",
    "rib_num",
    "aa_count_per_gap",
    "model_sequence",
    "sequence_molar_extinction_280",
    # "sequence_molecular_weight", # redundant to mass
    "backbone_loop_mask_string",
    "aligned_length",
    "seq_id",
    "sequence_name",
    "dssp_assignment",
    "aggrescan3d_avg_value",
    "hydrophobic_fitness",
    "packing_density",
    "rosetta_total_per_aa",
    "name",
    "overall_distance_score_v2",
    "overall_path_distance_score_v2",
    "overall_contact_order_score",
    "overall_linker_convenience_score",
    "overall_linker_convenience_v2_score",
    "path_score_v2",
    "path_score_v3",
]

composition_cols = [f"composition_{aa}" for aa in [
    "PHE", "TYR", "TRP", "GLY", "ALA", "VAL", "LEU", "ILE", "MET", "CYS",
    "PRO", "THR", "SER", "ASN", "GLN", "ASP", "GLU", "LYS", "ARG", "HIS"
]]

useful_cols = gene_synthesis_cols + metrics_cols + composition_cols

useful_cols_non_csv = useful_cols + [
    "taylor_letter_packing_descriptors",
    "chothia_omega_angles",
] 