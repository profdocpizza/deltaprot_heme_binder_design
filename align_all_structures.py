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
    cmd.reinitialize()  
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


def align_all_structures(pdb_path_df: pd.DataFrame,
                         output_dir: Path,):
    """
    For each row in pdb_path_df, align diffusion, fold_HEM, and fold_no_lig
    to the reference assembly, compute the requested RMSDs, and
    return a DataFrame with columns:
      [sequence_name, rmsd, rmsd_no_lig, rmsd_diffusion, rmsd_heme]
    """
    # sanity‐check input
    required = {
        "sequence_name",
        "assembly_pdb_path",
        "diffusion_pdb_path",
        "fold_HEM_pdb_path",
        "fold_no_lig_pdb_path"
    }
    assert required <= set(pdb_path_df.columns)

    records = []
    for _, row in tqdm(pdb_path_df.iterrows(), total=len(pdb_path_df)):
        seq = row["sequence_name"]

        # load
        asm   = ampal.load_pdb(row["assembly_pdb_path"])
        diff  = ampal.load_pdb(row["diffusion_pdb_path"])
        foldH = ampal.load_pdb(row["fold_HEM_pdb_path"])
        foldN = ampal.load_pdb(row["fold_no_lig_pdb_path"])

        # parse mask parameters from seq name
        parsed = parse_filename(seq.replace('_0',''))
        n_sel  = int(parsed["residues_per_helix"])
        n_skip = int(parsed["linker_length"])
        n_ribs = int(seq[1]) - 1
        mask   = generate_parametric_residue_mask(n_sel, n_skip, n_ribs)

        # CA coords
        ideal_Q = np.array([a.array for a in asm.get_atoms(ligands=False)
                             if a.res_label=='CA'])
        diff_P_all  = [a.array for a in diff.get_atoms(ligands=False)
                          if a.res_label=='CA']
        foldH_P_all = [a.array for a in foldH.get_atoms(ligands=False)
                          if a.res_label=='CA']
        foldN_P_all = [a.array for a in foldN.get_atoms(ligands=False)
                          if a.res_label=='CA']

        diff_P  = np.array([c for c,m in zip(diff_P_all,  mask) if m])
        foldH_P = np.array([c for c,m in zip(foldH_P_all, mask) if m])
        foldN_P = np.array([c for c,m in zip(foldN_P_all, mask) if m])

        # align each to the reference CA’s
        U_diff, T_diff, rmsd_diff  = align_P_to_Q_point_set(diff_P,  ideal_Q)
        U_fh,   T_fh,   rmsd_fh    = align_P_to_Q_point_set(foldH_P, ideal_Q)
        U_fn,   T_fn,   rmsd_fn    = align_P_to_Q_point_set(foldN_P, ideal_Q)

        # apply transforms in place
        apply_transform_to_assembly(diff,  U_diff,  T_diff,  translate_first=False)
        apply_transform_to_assembly(foldH, U_fh,    T_fh,    translate_first=False)
        apply_transform_to_assembly(foldN, U_fn,    T_fn,    translate_first=False)

        # extract HEM coords & compute flip‐ignorant RMSD
        atom_list     = ["FE","NA","NB","NC","ND"]
        atom_list_flip= ["FE","ND","NC","NB","NA"]

        ideal_L = extract_ligand_coords(asm,   mol_codes=["HEM"], atom_names=atom_list)
        fh_L    = extract_ligand_coords(foldH, mol_codes=["LIG"], atom_names=atom_list)
        fh_Lf   = extract_ligand_coords(foldH, mol_codes=["LIG"], atom_names=atom_list_flip)

        rmsd_heme      = calculate_rmsd(fh_L,  ideal_L)
        rmsd_heme_flip = calculate_rmsd(fh_Lf, ideal_L)
        rmsd_heme_min  = min(rmsd_heme, rmsd_heme_flip)
        
        rmsd_fe = np.linalg.norm(ideal_L[0] - fh_L[0])

        # optionally save a PyMOL session
        if rmsd_fh<1.5 and rmsd_fe<1.5:
            pdbs = {
                'reference': asm.make_pdb(ligands=False),
                'diffusion': diff.make_pdb(ligands=False),
                'fold_HEM' : foldH.make_pdb(ligands=False),
                'fold_no_lig': foldN.make_pdb(ligands=False),
            }
            pse_path = Path(output_dir)/ "aligned_pse" / f"{seq}.pse"
            os.makedirs(pse_path.parent, exist_ok=True)
            save_pymol_session(pdbs, str(pse_path), by_string=True)

        # collect results
        records.append({
            "sequence_name":  seq,
            "rmsd":           rmsd_fh,
            "rmsd_no_lig":    rmsd_fn,
            "rmsd_diffusion": rmsd_diff,
            "rmsd_heme":      rmsd_heme_min,
            "rmsd_fe":        rmsd_fe
        })
    
    # save as csv in output_dir / rmsd_df.csv
    df = pd.DataFrame.from_records(records,
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