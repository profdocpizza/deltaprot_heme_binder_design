import os
from pathlib import Path
import ampal
from dp_utils.helix_assembly.utils import align_P_to_Q_point_set,apply_transform_to_assembly, calculate_rmsd

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline_utils import parse_filename
from ampal.assembly import Assembly
from ampal.protein import Polypeptide
from pymol import cmd, finish_launching

# start PyMOL in headless mode (no GUI)
finish_launching(['pymol', '-qc'])  


def load_assembly(pdb_path: Path) -> Assembly:
    """
    Load a PDB file into an AMPAL assembly.
    """
    return ampal.load_pdb(str(pdb_path))


def get_ca_coordinates(assembly: Assembly) -> np.ndarray:
    """
    Extract all CA atom coordinates (as numpy array) from an AMPAL assembly.
    """
    return np.array([atom.array
                     for atom in assembly.get_atoms(ligands=False)
                     if atom.res_label == 'CA'])


def compute_alignment(mobile: np.ndarray,
                      reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Align a mobile point set to a reference point set.
    Returns the rotation matrix U, translation T, and RMSD.
    """
    return align_P_to_Q_point_set(mobile, reference)


def apply_alignment(assembly: Assembly,
                    rotation: np.ndarray,
                    translation: np.ndarray,
                    translate_first: bool = False) -> None:
    """
    Apply a rigid-body transformation to an AMPAL assembly.
    """
    apply_transform_to_assembly(assembly, rotation, translation,
                                translate_first=translate_first)



def compute_heme_rmsd(reference: Assembly,
                      folded: Assembly,
                      atom_names: list[str] = None) -> tuple[float, float]:
    """
    Compute the minimum heavy-atom RMSD for HEM ligands, accounting for possible flips,
    and the Fe-only distance.
    Returns (min_rmsd, fe_distance).
    """
    if atom_names is None:
        atom_names = ["FE", "NA", "NB", "NC", "ND"]
    flip = atom_names[::-1]

    ref_coords = extract_ligand_coords(reference, mol_codes=["HEM"], atom_names=atom_names)
    folded_coords = extract_ligand_coords(folded, mol_codes=["LIG"], atom_names=atom_names)
    folded_flip = extract_ligand_coords(folded, mol_codes=["LIG"], atom_names=flip)

    rmsd1 = calculate_rmsd(folded_coords, ref_coords)
    rmsd2 = calculate_rmsd(folded_flip, ref_coords)
    rmsd_min = min(rmsd1, rmsd2)

    fe_distance = np.linalg.norm(ref_coords[0] - folded_coords[0])
    return rmsd_min, fe_distance


def align_structures(assembly_path: Path,
                     diffusion_path: Path,
                     fold_HEM_path: Path,
                     fold_no_lig_path: Path,
                     sequence_name: str) -> tuple[dict, dict[str, Assembly]]:
    """
    Align diffusion, fold_HEM, and fold_no_lig assemblies to a reference.
    Returns a metrics dict and the loaded AMPAL assemblies.

    metrics keys: rmsd, rmsd_no_lig, rmsd_diffusion, rmsd_heme, rmsd_fe
    assemblies keys: reference, diffusion, fold_HEM, fold_no_lig
    """
    # Load assemblies
    ref_asm  = load_assembly(assembly_path)
    diff_asm = load_assembly(diffusion_path)
    fh_asm   = load_assembly(fold_HEM_path)
    fn_asm   = load_assembly(fold_no_lig_path)

    # Parse sequence parameters
    parsed = parse_filename(sequence_name.replace('_0', ''))
    n_sel  = int(parsed["residues_per_helix"])
    n_skip = int(parsed["linker_length"])
    n_ribs = int(sequence_name[1]) - 1
    mask   = generate_parametric_residue_mask(n_sel, n_skip, n_ribs)

    # Reference CA coords
    ref_ca = get_ca_coordinates(ref_asm)

    # Masked mobile coords helper
    def masked_coords(asm):
        all_ca = get_ca_coordinates(asm)
        return np.array([c for c, m in zip(all_ca, mask) if m])

    diff_ca = masked_coords(diff_asm)
    fh_ca   = masked_coords(fh_asm)
    fn_ca   = masked_coords(fn_asm)

    # Perform alignments
    U_diff, T_diff, rmsd_diff  = compute_alignment(diff_ca, ref_ca)
    U_fh,   T_fh,   rmsd_fh    = compute_alignment(fh_ca,   ref_ca)
    U_fn,   T_fn,   rmsd_fn    = compute_alignment(fn_ca,   ref_ca)

    # Apply transforms
    apply_alignment(diff_asm, U_diff, T_diff)
    apply_alignment(fh_asm,   U_fh,   T_fh)
    apply_alignment(fn_asm,   U_fn,   T_fn)

    # Compute HEM metrics
    rmsd_heme, rmsd_fe = compute_heme_rmsd(ref_asm, fh_asm)

    metrics = {
        "rmsd":           rmsd_fh,
        "rmsd_no_lig":    rmsd_fn,
        "rmsd_diffusion": rmsd_diff,
        "rmsd_heme":      rmsd_heme,
        "rmsd_fe":        rmsd_fe
    }

    return metrics, {
        'reference': ref_asm,
        'diffusion': diff_asm,
        'fold_HEM':  fh_asm,
        'fold_no_lig': fn_asm
    }


def align_all_structures(pdb_path_df: pd.DataFrame,
                         output_dir: Path) -> pd.DataFrame:
    """
    Process a DataFrame of PDB file paths, align all structures, save PDBs and sessions,
    and gather metrics.
    """
    required = {
        "sequence_name",
        "assembly_pdb_path",
        "diffusion_pdb_path",
        "fold_HEM_pdb_path",
        "fold_no_lig_pdb_path"
    }
    assert all([ col in pdb_path_df.columns for col in required]), "pdb_path_df is missing required columns: " + ", ".join(required)

    records = []
    session_dir = Path(output_dir) / "aligned_pse"
    session_dir.mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(pdb_path_df.iterrows(), total=len(pdb_path_df)):
        seq = row["sequence_name"]
        metrics, assemblies = align_structures(
            Path(row["assembly_pdb_path"]),
            Path(row["diffusion_pdb_path"]),
            Path(row["fold_HEM_pdb_path"]),
            Path(row["fold_no_lig_pdb_path"]),
            seq
        )

        # save session if within thresholds
        if metrics['rmsd'] < 1.5 and metrics['rmsd_fe'] < 1.5:
            session_dir.mkdir(parents=True, exist_ok=True)
            pse_path = session_dir / f"{seq}.pse"
            pdbs = {name: asm.make_pdb() for name, asm in assemblies.items()}
            save_pymol_session(pdbs, str(pse_path), by_string=True)

        metrics["sequence_name"] = seq
        records.append(metrics)

    df = pd.DataFrame(records,
                      columns=[
                          "sequence_name",
                          "rmsd",
                          "rmsd_no_lig",
                          "rmsd_diffusion",
                          "rmsd_heme",
                          "rmsd_fe"
                      ])
    df.to_csv(Path(output_dir) / "rmsd_df.csv", index=False)
    return df




####################### OLD CODE ########################

# import os
# from pathlib import Path
# import ampal
# from dp_utils.helix_assembly.utils import align_P_to_Q_point_set,apply_transform_to_assembly, calculate_rmsd

# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# from pipeline_utils import parse_filename
# from ampal.assembly import Assembly
# from ampal.protein import Polypeptide
# from pymol import cmd, finish_launching

# # start PyMOL in headless mode (no GUI)
# finish_launching(['pymol', '-qc'])  


def generate_design_df(assemblies_dir, diffusion_dir, fold_HEM_dir_root, fold_no_lig_dir_root, save_dir):
    records = []
    for fold_HEM_dir in tqdm(Path(fold_HEM_dir_root).iterdir()):
        sequence_name = fold_HEM_dir.name

        assembly_pdb_path = Path(assemblies_dir) / f"{sequence_name.replace('_0', '')}.pdb"
        diffusion_pdb_path = Path(diffusion_dir) / "outputs" / f"{sequence_name}.pdb"
        fold_HEM_pdb_path = Path(fold_HEM_dir_root) /  sequence_name / f"{sequence_name}_model_0.pdb"
        fold_no_lig_pdb_path = Path(fold_no_lig_dir_root) / sequence_name / f"{sequence_name}_model_0.pdb"
        assert assembly_pdb_path.exists(), f"assembly_pdb_path {assembly_pdb_path} does not exist"
        assert diffusion_pdb_path.exists(), f"diffusion_pdb_path {diffusion_pdb_path} does not exist"
        assert fold_HEM_pdb_path.exists(), f"fold_HEM_pdb_path {fold_HEM_pdb_path} does not exist"
        # assert fold_no_lig_pdb_path.exists(), f"fold_no_lig_pdb_path {fold_no_lig_pdb_path} does not exist"
        # Append a record for this sequence
        records.append({
            'sequence_name': sequence_name,
            'assembly_pdb_path': str(assembly_pdb_path),
            'diffusion_pdb_path': str(diffusion_pdb_path),
            'fold_HEM_pdb_path': str(fold_HEM_pdb_path),
            'fold_no_lig_pdb_path': str(fold_no_lig_pdb_path)
        })

    # Create DataFrame from collected records
    df = pd.DataFrame(records)
    # save as csv in save_dir / design_df.csv
    df.to_csv(Path(save_dir) / "design_df.csv", index=False)
    return df

def generate_parametric_residue_mask(n_select, n_skip, n_ribs):
    """
    Generate a mask of length:
      n_ribs*n_select + (n_ribs-1)*n_skip

    Pattern:
      [True]*n_select,
      [False]*n_skip,
      [True]*n_select,
      …
      [True]*n_select    # last block, no skip after
    """
    mask = []
    for rib in range(n_ribs):
        # select block
        mask.extend([True]  * n_select)
        # skip block (except after the final rib)
        if rib < n_ribs - 1:
            mask.extend([False] * n_skip)
    return mask


def extract_ligand_coords(asm, mol_codes,atom_names):
    """
    Grab all atom coordinates for every HEM ligand in `asm`.
    Returns an (N,3) NumPy array.
    """
    # get_ligands(solvent=False) returns all non‐solvent Monomers
    coords = []
    # loop over every ligand‐group in the assembly
    for lig in asm.get_ligands(solvent=False):
        if lig.mol_code in mol_codes:
            # ensure we get one block of len(atom_names) per ligand
            for atom_name in atom_names:
                atom = lig.atoms.get(atom_name)
                if atom is None:
                    raise ValueError(
                        f"Ligand {lig.mol_code} (chain {lig.id!r}) "
                        f"is missing atom '{atom_name}'"
                    )
                coords.append(atom.array)

    return np.array(coords)

def save_pymol_session(
    objects: dict[str, str],
    output_path: str,
    by_string: bool = True
):
    """
    Load multiple structures into PyMOL and save a .pse session.

    Args:
      objects    : mapping of object_name -> pdb_string (or pdb_path if by_string False)
      output_path: where to write the .pse file
      by_string  : if True, treat each value as a PDB-format string;
                   if False, treat each value as a filename
    """
    for name, data in objects.items():
        if by_string:
            cmd.read_pdbstr(data, name)
        else:
            cmd.load(data, name)
    # optional: set a nice default view
    cmd.bg_color('white')
    cmd.orient()
    cmd.save(output_path)
    cmd.delete('all')  # clear for next run

import ProteinStructureInspector

def save_pymol_session(
    objects: dict[str, str],
    output_path: str,
    by_string: bool = True,
    morph: bool = True,
):
    """
    Load multiple structures into PyMOL, optionally perform a morph between fold_HEM and fold_no_lig,
    and save a .pse session.

    Args:
      objects       : mapping of object_name -> pdb_string (or pdb_path if by_string False)
      output_path   : where to write the .pse file
      by_string     : if True, treat each value as a PDB-format string;
                      if False, treat each value as a filename
      morph         : whether to perform a morph between fold_HEM and fold_no_lig
      morph_name    : name of the morph object to create
      morph_frames  : number of frames in the morph
    """
    # clear any existing session
    cmd.reinitialize()

    # load each object
    for name, data in objects.items():
        if by_string:
            cmd.read_pdbstr(data, name)
        else:
            cmd.load(data, name)

    # optional: perform morph if requested
    if morph:
        # ensure both required objects exist
        if 'fold_HEM' not in objects or 'fold_no_lig' not in objects:
            raise ValueError("Both 'fold_HEM' and 'fold_no_lig' must be provided in objects for morphing")
        # morph between fold_HEM and fold_no_lig
        cmd.morph("fold_HEM_to_fold_no_lig", 'fold_HEM', 'fold_no_lig')
        cmd.show('cartoon', "fold_HEM_to_fold_no_lig")
        cmd.morph(
            'diffusion_to_fold_HEM',
            'diffusion',
            'fold_HEM',
        )
        cmd.show('cartoon', 'diffusion_to_fold_HEM')

    # set a nice default view
    cmd.bg_color('white')
    cmd.orient()
    ProteinStructureInspector.main()
    cmd.disable('all')
    cmd.enable('fold_HEM')
    cmd.enable('fold_HEM_to_fold_no_lig')
    cmd.save(output_path)

    # clear for next run
    cmd.delete('all')

# def align_all_structures(pdb_path_df: pd.DataFrame,
#                          output_dir: Path,):
#     """
#     For each row in pdb_path_df, align diffusion, fold_HEM, and fold_no_lig
#     to the reference assembly, compute the requested RMSDs, and
#     return a DataFrame with columns:
#       [sequence_name, rmsd, rmsd_no_lig, rmsd_diffusion, rmsd_heme]
#     """
#     # sanity‐check input
#     required = {
#         "sequence_name",
#         "assembly_pdb_path",
#         "diffusion_pdb_path",
#         "fold_HEM_pdb_path",
#         "fold_no_lig_pdb_path"
#     }
#     assert required <= set(pdb_path_df.columns)

#     records = []
#     for _, row in tqdm(pdb_path_df.iterrows(), total=len(pdb_path_df)):
#         seq = row["sequence_name"]
#         asm   = ampal.load_pdb(row["assembly_pdb_path"])
#         diff  = ampal.load_pdb(row["diffusion_pdb_path"])
#         foldH = ampal.load_pdb(row["fold_HEM_pdb_path"])
#         foldN = ampal.load_pdb(row["fold_no_lig_pdb_path"])

#         # parse mask parameters from seq name
#         parsed = parse_filename(seq.replace('_0',''))
#         n_sel  = int(parsed["residues_per_helix"])
#         n_skip = int(parsed["linker_length"])
#         n_ribs = int(seq[1]) - 1
#         mask   = generate_parametric_residue_mask(n_sel, n_skip, n_ribs)

#         # CA coords
#         ideal_Q = np.array([a.array for a in asm.get_atoms(ligands=False)
#                              if a.res_label=='CA'])
#         diff_P_all  = [a.array for a in diff.get_atoms(ligands=False)
#                           if a.res_label=='CA']
#         foldH_P_all = [a.array for a in foldH.get_atoms(ligands=False)
#                           if a.res_label=='CA']
#         foldN_P_all = [a.array for a in foldN.get_atoms(ligands=False)
#                           if a.res_label=='CA']

#         diff_P  = np.array([c for c,m in zip(diff_P_all,  mask) if m])
#         foldH_P = np.array([c for c,m in zip(foldH_P_all, mask) if m])
#         foldN_P = np.array([c for c,m in zip(foldN_P_all, mask) if m])

#         # align each to the reference CA’s
#         U_diff, T_diff, rmsd_diff  = align_P_to_Q_point_set(diff_P,  ideal_Q)
#         U_fh,   T_fh,   rmsd_fh    = align_P_to_Q_point_set(foldH_P, ideal_Q)
#         U_fn,   T_fn,   rmsd_fn    = align_P_to_Q_point_set(foldN_P, ideal_Q)

#         # apply transforms in place
#         apply_transform_to_assembly(diff,  U_diff,  T_diff,  translate_first=False)
#         apply_transform_to_assembly(foldH, U_fh,    T_fh,    translate_first=False)
#         apply_transform_to_assembly(foldN, U_fn,    T_fn,    translate_first=False)

#         # extract HEM coords & compute flip‐ignorant RMSD
#         atom_list     = ["FE","NA","NB","NC","ND"]
#         atom_list_flip= ["FE","ND","NC","NB","NA"]

#         ideal_L = extract_ligand_coords(asm,   mol_codes=["HEM"], atom_names=atom_list)
#         fh_L    = extract_ligand_coords(foldH, mol_codes=["LIG"], atom_names=atom_list)
#         fh_Lf   = extract_ligand_coords(foldH, mol_codes=["LIG"], atom_names=atom_list_flip)

#         rmsd_heme      = calculate_rmsd(fh_L,  ideal_L)
#         rmsd_heme_flip = calculate_rmsd(fh_Lf, ideal_L)
#         rmsd_heme_min  = min(rmsd_heme, rmsd_heme_flip)
        
#         rmsd_fe = np.linalg.norm(ideal_L[0] - fh_L[0])

#         # optionally save a PyMOL session
#         if rmsd_fh<1.5 and rmsd_fe<1.5:
#             pdbs = {
#                 'reference': asm.make_pdb(),
#                 'diffusion': diff.make_pdb(),
#                 'fold_HEM' : foldH.make_pdb(),
#                 'fold_no_lig': foldN.make_pdb(),
#             }
#             pse_path = Path(output_dir)/ "aligned_pse" / f"{seq}.pse"
#             os.makedirs(pse_path.parent, exist_ok=True)
#             save_pymol_session(pdbs, str(pse_path), by_string=True)

#         # collect results
#         records.append({
#             "sequence_name":  seq,
#             "rmsd":           rmsd_fh,
#             "rmsd_no_lig":    rmsd_fn,
#             "rmsd_diffusion": rmsd_diff,
#             "rmsd_heme":      rmsd_heme_min,
#             "rmsd_fe":        rmsd_fe
#         })
    
#     # save as csv in output_dir / rmsd_df.csv
#     df = pd.DataFrame.from_records(records,
#                                    columns=[
#                                         "sequence_name",
#                                         "rmsd",
#                                         "rmsd_no_lig",
#                                         "rmsd_diffusion",
#                                         "rmsd_heme",
#                                         "rmsd_fe"
#                                      ])
#     df.to_csv(Path(output_dir) / "rmsd_df.csv", index=False)

#     return df