import json
import shutil
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from ampal.pdb_parser import load_pdb
from ampal.assembly import AmpalContainer
import random
from typing import Dict, Union, List
import ampal.geometry as geom
from dp_utils.helix_assembly.hydrophobic_core import add_cb_atoms
from scipy.fft import set_backend
from tqdm import tqdm
import os
import ampal

from align_all_structures import generate_parametric_residue_mask
from pipeline_utils import parse_filename
# -------------------------------------
# Constants / Paths
# -------------------------------------
# Root directory containing per-model folders
ROOT_DIR = Path(
    "/home/tadas/code/deltaprot_heme_binder_design/outputs/4_sequence_prediction/outputs"
)
# Path to write selected sequences
SELECTED_FASTA = (
    Path("/home/tadas/code/deltaprot_heme_binder_design/outputs/4_sequence_prediction")
    / "selected_sequences.fasta"
)



# def validate_liganmpnn_heme_binder_sequence(
#     pdb_file: Path,
#     his_fe_cutoff: float = 5.0,
#     cationic_heme_anionic_cutoff: float = 5.0,
# ) -> bool:
#     """
#     Validate a heme-binding protein sequence.

#     Checks:
#     - Sequence contains at least one histidine, one positive residue (Arg/Lys), and one chromophore (Tyr/Trp).
#     - At least one His atom (NE2 or ND1) is within `his_fe_cutoff` Å of any Fe atom.
#     - At least one Arg/Lys atom (NH1, NH2, or NZ) is within `cationic_heme_anionic_cutoff` Å of any heme oxygen atom (O1A, O1D, O2A, O2D).

#     Returns True if all checks pass, False otherwise.
#     """
#     try:
#         assembly = load_pdb(str(pdb_file), path=True)
#     except Exception as e:
#         print(f"[ERROR] Failed to load {pdb_file}: {e}")
#         return False

#     # Unwrap AmpalContainer if needed
#     if hasattr(assembly, "ampal_objects"):
#         assembly = assembly.ampal_objects[0]

#     # Sequence composition checks
#     seq = assembly.sequences[0]
#     if not (
#         "H" in seq
#         and any(res in seq for res in ("R", "K"))
#         and any(res in seq for res in ("Y", "W"))
#     ):
#         print(
#             f"Sequence composition failed: HIS={'H' in seq}, POS={any(res in seq for res in ('R','K'))}, CHROMO={any(res in seq for res in ('Y','W'))}"
#         )
#         return False

#     # Prepare atom sets
#     his_atom_names = ("NE2", "ND1")
#     pos_atom_map = {"ARG": ("NH1", "NH2"), "LYS": ("NZ",)}
#     oxy_names = ("O1A", "O1D", "O2A", "O2D")

#     # Fetch atoms and monomers
#     fe_atoms = [
#         atom
#         for atom in assembly.get_atoms(ligands=True)
#         if atom.element.upper() == "FE"
#     ]
#     his_residues = [
#         res for res in assembly.get_monomers(ligands=False) if res.mol_code == "HIS"
#     ]
#     pos_residues = [
#         res
#         for res in assembly.get_monomers(ligands=False)
#         if res.mol_code in pos_atom_map
#     ]
#     heme_monomers = [
#         mon for mon in assembly.get_monomers(ligands=True) if mon.mol_code == "HEM"
#     ]

#     # Helper to get Atom list from residue (dict or OrderedDict or list)
#     def get_atoms_list(residue):
#         atoms_attr = getattr(residue, "atoms", None)
#         if isinstance(atoms_attr, dict):
#             return atoms_attr.values()
#         return atoms_attr or []

#     # Check His-Fe proximity
#     if fe_atoms and his_residues:
#         cutoff2_his = his_fe_cutoff**2
#         for fe in fe_atoms:
#             for his_res in his_residues:
#                 for atom in get_atoms_list(his_res):
#                     if (
#                         atom.res_label in his_atom_names
#                         and np.sum((atom.array - fe.array) ** 2) <= cutoff2_his
#                     ):
#                         break
#                 else:
#                     continue
#                 break
#             else:
#                 continue
#             break
#         else:
#             return False

#     # Check cationic-heme anionic proximity
#     if heme_monomers and pos_residues:
#         cutoff2_cat = cationic_heme_anionic_cutoff**2
#         for heme in heme_monomers:
#             for oxy_name in oxy_names:
#                 oxy_atom = heme.atoms.get(oxy_name)
#                 if oxy_atom is None:
#                     continue
#                 for pos_res in pos_residues:
#                     for atom in get_atoms_list(pos_res):
#                         if (
#                             atom.res_label in pos_atom_map[pos_res.mol_code]
#                             and np.sum((atom.array - oxy_atom.array) ** 2)
#                             <= cutoff2_cat
#                         ):
#                             break
#                     else:
#                         continue
#                     break
#                 else:
#                     continue
#                 break
#             else:
#                 return False

#     return True

def validate_liganmpnn_heme_binder_sequence(
    pdb_file: Path,
    his_fe_cutoff: float = 4.2,
    cationic_heme_anionic_cutoff: float = 5.0,
) -> bool:
    """
    Validate a heme‐binding design by sequence composition, spacing, and structural proximity.
    """
    try:
        assembly = load_pdb(str(pdb_file), path=True)
    except Exception as e:
        print(f"[ERROR] Failed to load {pdb_file}: {e}")
        return False

    if isinstance(assembly, AmpalContainer):
        assembly = assembly[0]

    seq = assembly.sequences[0]

    # 1) Histidine count
    his_count = seq.count('H')
    if his_count not in (1, 2):
        print(f"Sequence composition failed: expected 1 or 2 histidines, found {his_count}")
        return False

    # 2) Positive residue check
    if not any(c in seq for c in ('R', 'K')):
        print("Sequence composition failed: need at least one positively charged residue (R/K)")
        return False

    # 3) Chromophore check
    if not any(c in seq for c in ('Y', 'W')):
        print("Sequence composition failed: need at least one chromophore residue (Y/W)")
        return False

    # 4) No double‐His too close
    if has_double_his(seq, max_gap=5):
        print("Double HIS check failed: two histidines <=5 residues apart")
        return False

    # 5) Proximity checks
    if not check_heme_proximity(assembly, his_fe_cutoff, cationic_heme_anionic_cutoff):
        print("Proximity checks failed: His–Fe or cationic–heme not satisfied")
        return False

    return True


def check_heme_proximity(
    assembly,
    his_fe_cutoff: float = 5.0,
    cationic_heme_anionic_cutoff: float = 5.0
) -> bool:
    """
    Perform proximity checks:
      1) Ensure ALL His residues (NE2/ND1) are within his_fe_cutoff Å of at least one Fe atom.
      2) At least one Arg/Lys (NH1/NH2/NZ) is within cationic_heme_anionic_cutoff Å of a HEM oxygen atom.
    Returns True if both checks pass.
    """
    # Atom/residue definitions
    his_names = ('NE2', 'ND1')
    pos_map = {'ARG': ('NH1', 'NH2'), 'LYS': ('NZ',)}
    oxy_names = ('O1A', 'O1D', 'O2A', 'O2D')

    # Collect atoms and residues
    fe_atoms = [at for at in assembly.get_atoms(ligands=True) if at.element.upper() == 'FE']
    his_residues = [r for r in assembly.get_monomers(ligands=False) if r.mol_code == 'HIS']
    pos_residues = [r for r in assembly.get_monomers(ligands=False) if r.mol_code in pos_map]
    heme_monomers = [m for m in assembly.get_monomers(ligands=True) if m.mol_code == 'HEM']

    def atoms_of(res):
        attr = getattr(res, 'atoms', None)
        if isinstance(attr, dict):
            return attr.values()
        return attr or []

    # 1) All His must be proximal to Fe
    if fe_atoms and his_residues:
        cutoff2 = his_fe_cutoff ** 2
        for his in his_residues:
            # check if this his has any atom within cutoff to any Fe
            proximal = False
            for atom in atoms_of(his):
                if atom.res_label in his_names:
                    for fe in fe_atoms:
                        if np.sum((atom.array - fe.array) ** 2) <= cutoff2:
                            proximal = True
                            break
                    if proximal:
                        break
            if not proximal:
                return False

    # 2) Cationic–heme oxygen: at least one Arg/Lys proximal to any heme oxygen
    if heme_monomers and pos_residues:
        cutoff2 = cationic_heme_anionic_cutoff ** 2
        found_pair = False
        for heme in heme_monomers:
            for oxy_nm in oxy_names:
                oxy = heme.atoms.get(oxy_nm)
                if oxy is None:
                    continue
                for pos in pos_residues:
                    for atom in atoms_of(pos):
                        if atom.res_label in pos_map[pos.mol_code] \
                           and np.sum((atom.array - oxy.array) ** 2) <= cutoff2:
                            found_pair = True
                            break
                    if found_pair:
                        break
                if found_pair:
                    break
            if found_pair:
                break
        if not found_pair:
            return False

    return True

def has_double_his(seq: str, max_gap: int = 5) -> bool:
    """
    Check for two histidines separated by no more than `max_gap` other residues.
    Returns True if such a pair exists.
    """
    positions = [i for i, aa in enumerate(seq) if aa == 'H']
    for i in range(len(positions) - 1):
        if (positions[i + 1] - positions[i] - 1) <= max_gap:
            return True
    return False


def find_residues_near_fe(
    pdb_file: Union[str, Path],
    min_dist: float = 3.0,
    max_dist: float = 7.0,
    ca_cb_angle_threshold: float = 90.0,
) -> List[str]:
    """
    Load the PDB at pdb_file, idealize to add CB atoms,
    find all FE atoms, then return a sorted list of unique residues
    (chain + residue number) for which:
      1) the CB atom is between `min_dist` and `max_dist` Å of any FE atom, and
      2) the angle between CA->CB and CA->Fe vectors is < ca_cb_angle_threshold.

    Returns:
        List of strings like ["A11", "C12", ...]
    """
    try:
        assembly = load_pdb(str(pdb_file), path=True)
    except Exception as e:
        print(f"[ERROR] Cannot load {pdb_file}: {e}")
        return []

    if isinstance(assembly, AmpalContainer):
        assembly = assembly[0]


    # ensure CB atoms exist
    assembly = add_cb_atoms(assembly)

    # collect all Fe atoms
    fe_atoms = [atom for atom in assembly.get_atoms() if atom.element.upper() == "FE"]
    if not fe_atoms:
        return []

    # collect all protein residues
    residues = assembly.get_monomers(ligands=False)
    nearby = set()

    for fe in fe_atoms:
        for res in residues:
            ca = res.atoms.get("CA")
            cb = res.atoms.get("CB")
            if ca is None or cb is None:
                continue
            # distance filter on CB
            d_cb_fe = geom.distance(cb, fe)
            if d_cb_fe < min_dist or d_cb_fe > max_dist:
                continue
            # angle filter
            v_ca_cb = np.subtract(cb.array, ca.array)
            v_ca_fe = np.subtract(fe.array, ca.array)
            dot = np.dot(v_ca_cb, v_ca_fe)
            norm_prod = np.linalg.norm(v_ca_cb) * np.linalg.norm(v_ca_fe)
            if norm_prod == 0:
                continue
            angle = np.degrees(np.arccos(dot / norm_prod))
            if angle >= ca_cb_angle_threshold:
                continue
            # passed filters
            chain_id = res.parent.id
            res_id = res.id
            nearby.add(f"{chain_id}{res_id}")

    # sort by chain then residue number
    def _sort_key(entry: str):
        chain = entry[0]
        num = int(entry[1:])
        return (chain, num)

    return sorted(nearby, key=_sort_key)


# -------------------------------------
# Process one folder
# -------------------------------------
# def process_folder(folder: Path):
#     packed_dir = folder / "packed"
#     seqs_dir = folder / "seqs"
#     if not packed_dir.exists() or not seqs_dir.exists():
#         return None

#     # scan sorted PDB names
#     for pdb_path in sorted(packed_dir.glob("*.pdb")):
#         if not validate_liganmpnn_heme_binder_sequence(pdb_path):
#             continue
#         # extract sequence ID from PDB filename suffix
#         parts = pdb_path.stem.split("_")
#         seq_id = parts[-1]

#         # find and parse multi-FASTA
#         fasta_file = next(seqs_dir.glob("*.fa"), None)
#         if not fasta_file:
#             return None
#         text = fasta_file.read_text()
#         # split records and find matching id
#         records = [r for r in text.split(">") if r.strip()]
#         for rec in records:
#             lines = rec.splitlines()
#             header = lines[0]
#             if f"id={seq_id}" in header:
#                 seq = "".join(lines[1:])
#                 return f">{folder.name}", seq, pdb_path
#         # if no matching record, skip to next PDB
#     return None

def process_folder(folder: Path):
    packed_dir = folder / "packed"
    if not packed_dir.exists():
        return None

    # scan sorted PDB names
    for pdb_path in sorted(packed_dir.glob("*.pdb")):
        if not validate_liganmpnn_heme_binder_sequence(pdb_path):
            continue

        # Load the assembly and extract the sequence directly
        assembly = load_pdb(str(pdb_path), path=True)
        if hasattr(assembly, "ampal_objects"):
            assembly = assembly.ampal_objects[0]
        seq = assembly.sequences[0]

        # Return FASTA header (folder name) and the sequence
        return f">{folder.name}", seq, pdb_path

    return None

# -------------------------------------
# Main
# -------------------------------------
# def select_liganmpnn_sequences(
#     ligandmpnn_root_dir, selected_fasta_path, selected_packed_structures_dir
# ):
#     ligandmpnn_root_dir = Path(ligandmpnn_root_dir)
#     selected_fasta_path = Path(selected_fasta_path)
#     selected_packed_structures_dir = Path(selected_packed_structures_dir)

#     selected_packed_structures_dir.mkdir(exist_ok=True)

#     folders = [d for d in sorted(ligandmpnn_root_dir.iterdir()) if d.is_dir()]
#     entries = []

#     with ProcessPoolExecutor() as executor:
#         futures = {executor.submit(process_folder, f): f for f in folders}
#         for future in as_completed(futures):
#             result = future.result()
#             if result:
#                 entries.append(result)

#     # write in one pass
#     selected_fasta_path.write_text("")
#     with selected_fasta_path.open("a") as out_f:
#         for header, seq, pdb_path in entries:
#             out_f.write(f"{header}\n{seq}\n")
#             # shutil copy pdb path to ligandmpnn_root_dir/selected_sequences_packed folder
#             shutil.copy(pdb_path, selected_packed_structures_dir / pdb_path.name)
def select_liganmpnn_sequences(
    ligandmpnn_root_dir, selected_fasta_path, selected_packed_structures_dir
):
    ligandmpnn_root_dir = Path(ligandmpnn_root_dir)
    selected_fasta_path = Path(selected_fasta_path)
    selected_packed_structures_dir = Path(selected_packed_structures_dir)

    selected_packed_structures_dir.mkdir(exist_ok=True, parents=True)
    folders = [d for d in sorted(ligandmpnn_root_dir.iterdir()) if d.is_dir()]
    entries = []

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_folder, f): f for f in folders}
        for future in as_completed(futures):
            result = future.result()
            if result:
                entries.append(result)

    # Write selected sequences to FASTA and copy PDBs
    with selected_fasta_path.open("w") as out_f:
        for header, seq, pdb_path in entries:
            out_f.write(f"{header}\n{seq}\n")
            shutil.copy(pdb_path, selected_packed_structures_dir / pdb_path.name)


# def make_ligand_mpnn_script(
#     input_dir: str,
#     output_dir: str,
#     run_py: str = "run.py",
#     model_type: str = "ligand_mpnn",
#     seed: int = 111,
#     num_seq_per_target: int = 500,
#     batch_size: int = 50,
#     number_of_batches: int = None,
#     temperature: float = None,
#     checkpoint: str = None,
#     gpu_devices: str = None,
# ) -> str:
#     """
#     Create a bash script that runs LigandMPNN (via run.py) on all PDBs in input_dir,
#     writing outputs to matching subdirectories in output_dir. Does not pack side chains.

#     If `gpu_devices` is provided (e.g. "0,1"), sets CUDA_VISIBLE_DEVICES accordingly.
#     Returns the path to the generated script.
#     """
#     input_dir = Path(input_dir)
#     script_path = Path(output_dir) / "run_ligand_mpnn.sh"
#     output_dir = Path(output_dir) / "outputs"
#     os.makedirs(output_dir, exist_ok=True)

#     with open(script_path, "w") as fh:
#         fh.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
#         # GPU restriction
#         if gpu_devices:
#             fh.write(f"export CUDA_VISIBLE_DEVICES={gpu_devices}\n\n")
#         fh.write(
#             f'RUN_PY="{run_py}"\nMODEL_TYPE="{model_type}"\nSEED={seed}\nNUM_SEQ={num_seq_per_target}\nBATCH={batch_size}\n'
#         )
#         if number_of_batches:
#             fh.write(f"NUM_BATCHES={number_of_batches}\n")
#         if temperature is not None:
#             fh.write(f"TEMP={temperature}\n")
#         if checkpoint:
#             cp = Path(checkpoint)
#             fh.write(f'CHECKPOINT="{cp}"\n')
#         fh.write('\nmkdir -p "' + str(output_dir) + '"\n')
#         fh.write(f"for pdb in $(shuf -e {input_dir}/*.pdb); do\n")
#         fh.write("  base=$(basename $pdb .pdb)\n")
#         fh.write('  outdir="' + str(output_dir) + '/$base"\n')
#         fh.write("  mkdir -p $outdir\n")
#         # build command
#         cmd = [
#             "python",
#             "$RUN_PY",
#             "--model_type",
#             "$MODEL_TYPE",
#             "--seed",
#             "$SEED",
#             "--pdb_path",
#             "$pdb",
#             "--out_folder",
#             "$outdir",
#             "--batch_size",
#             "$BATCH",
#         ]
#         if number_of_batches:
#             cmd += ["--number_of_batches", "$NUM_BATCHES"]
#         if temperature is not None:
#             cmd += ["--temperature", "$TEMP"]
#         if checkpoint:
#             cmd += ["--checkpoint_ligand_mpnn", "$CHECKPOINT"]
#         cmd += ["--pack_side_chains", "1"]
#         cmd += ["--number_of_packs_per_design", "1"]
#         cmd += ["--pack_with_ligand_context", "1"]
#         # cmd += ["--apply_custom_filter", "1"]
#         cmd += ["--omit_AA", "C"]
#         cmd += [
#             "--bias_AA",
#             "H:1,Y:1,W:1,A:-1.6",
#         ]  # TODO: try "H:1.5,Y:1.5,W:1.5,A:-1.6,G:-1"] to get more positives
#         # write command
#         fh.write("  " + " \\\n    ".join(cmd) + "\n")
#         fh.write("done\n")

#     os.chmod(script_path, 0o755)
#     return str(script_path)


# def make_ligand_mpnn_script(
#     input_dir: Union[str, Path],
#     output_dir: Union[str, Path],
#     run_py: str = "run.py",
#     model_type: str = "ligand_mpnn",
#     seed: int = 111,
#     num_seq_per_target: int = 500,
#     batch_size: int = 50,
#     number_of_batches: int = None,
#     temperature: float = None,
#     checkpoint: str = None,
#     gpu_devices: str = None,
#     his_bias: float = 3.0,
#     min_dist: float = 3.0,
#     max_dist: float = 7.0,
#     ca_cb_angle_threshold: float = 90.0,
# ) -> str:
#     """
#     Generate a bash script that runs LigandMPNN on each PDB (one line per pdb):
#       - computes nearby residues via find_residues_near_fe
#       - writes a per-pdb bias JSON mapping each residue to {"H": his_bias, "K": pos_bias, "R": pos_bias}
#       - calls run.py with both global bias_AA for Y/W/A and per-residue biases
#     """
#     input_dir = Path(input_dir)
#     output_dir = Path(output_dir)
#     script_path = output_dir / "run_ligand_mpnn.sh"
#     bias_dir = output_dir / "custom_biases"
#     omit_dir = output_dir / "custom_omits"
#     os.makedirs(bias_dir, exist_ok=True)
#     os.makedirs(omit_dir, exist_ok=True)
#     os.makedirs(output_dir / "outputs", exist_ok=True)


#     with open(script_path, "w") as fh:
#         # header variables
#         fh.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
#         if gpu_devices:
#             fh.write(f"export CUDA_VISIBLE_DEVICES={gpu_devices}\n\n")
#         fh.write(
#             f'RUN_PY="{run_py}"\n'
#             f'MODEL_TYPE="{model_type}"\n'
#             f"SEED={seed}\n"
#             f"NUM_SEQ={num_seq_per_target}\n"
#             f"BATCH={batch_size}\n"
#         )
#         if number_of_batches:
#             fh.write(f"NUM_BATCHES={number_of_batches}\n")
#         if temperature is not None:
#             fh.write(f"TEMP={temperature}\n")
#         if checkpoint:
#             fh.write(f'CHECKPOINT="{checkpoint}"\n')
#         # global bias for all lines
#         fh.write('GLOBAL_BIAS_AA="Y:1.3,W:1.3,A:-1.3,G:-1.3"\n')
#         fh.write('GLOBAL_OMIT_AA="XC"\n\n')
#         fh.write(f'BIAS_DIR="{bias_dir}"\n')
#         fh.write(f'OMIT_DIR="{omit_dir}"\n\n')

#         pdb_list = list(input_dir.glob("*.pdb"))
#         random.shuffle(pdb_list)
#         for pdb_path in tqdm(pdb_list):
#             base = pdb_path.stem
#             outdir = output_dir / "outputs" / base
#             fh.write(f'mkdir -p "{outdir}"\n')


#             # compute and save per-residue bias JSON
#             residues = find_residues_near_fe(
#                 pdb_path, min_dist, max_dist, ca_cb_angle_threshold
#             )
#             bias_map: Dict[str, Dict[str, float]] = {}
#             for res in residues:
#                 bias_map[res] = {"H": his_bias}
#             bias_file = bias_dir / f"{base}.json"
#             with open(bias_file, "w") as bf:
#                 json.dump(bias_map, bf)

#             # --- per-residue omit JSON (forbid G where mask==True) ---
#             parsed = parse_filename(os.path.basename(pdb_path).replace('_0.pdb', ''))
#             n_sel  = int(parsed["residues_per_helix"])
#             n_skip = int(parsed["linker_length"])
#             n_ribs = int(parsed["orientation_code"][1]) - 1
#             parametric_res_mask = generate_parametric_residue_mask(n_sel, n_skip, n_ribs,diffuse_termini=True)
#             # parametric_res_mask = [False, False, True, True, True, True, True, True, True, ...]
#             omit_map: Dict[str, List[str]] = {}
#             for idx, omit_flag in enumerate(parametric_res_mask):
#                 if omit_flag:
#                     omit_map[idx+1] = ["G"]  
#             omit_file = omit_dir / f"{base}.json"
#             with open(omit_file, "w") as of:
#                 json.dump(omit_map, of)

#             # write single-line run command
#             cmd = [
#                 "python $RUN_PY",
#                 "--model_type $MODEL_TYPE",
#                 "--seed $SEED",
#                 f'--pdb_path "{pdb_path}"',
#                 f"--bias_AA $GLOBAL_BIAS_AA",
#                 f"--omit_AA $GLOBAL_OMIT_AA",
#                 f"--bias_AA_per_residue $BIAS_DIR/{base}.json",
#                 f"--omit_AA_per_residue $OMIT_DIR/{base}.json",
#                 f'--out_folder "{outdir}"',
#                 "--batch_size $BATCH",
#             ]
#             if number_of_batches:
#                 cmd.append("--number_of_batches $NUM_BATCHES")
#             if temperature is not None:
#                 cmd.append("--temperature $TEMP")
#             if checkpoint:
#                 cmd.append("--checkpoint_ligand_mpnn $CHECKPOINT")

#             # fixed options
#             cmd.extend(
#                 [
#                     "--pack_side_chains 1",
#                     "--number_of_packs_per_design 1",
#                     "--pack_with_ligand_context 1",
#                 ]
#             )

#             fh.write(" ".join(cmd) + "\n\n")

#     os.chmod(script_path, 0o755)
#     return str(script_path)


def make_ligand_mpnn_script(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    run_py: str = "run.py",
    model_type: str = "ligand_mpnn",
    seed: int = 111,
    num_seq_per_target: int = 500,
    batch_size: int = 50,
    number_of_batches: int = None,
    temperature: float = None,
    checkpoint: str = None,
    gpu_devices: str = None,
    his_bias: float = 2.0,
    min_dist: float = 3.0,
    max_dist: float = 7.0,
    ca_cb_angle_threshold: float = 90.0,
) -> str:
    """
    For each PDB in input_dir:
      1) load with ampal → get an ordered list of ALL designable residues
      2) call find_residues_near_fe() for the HIS bias subset
      3) generate parametric_res_mask of same length
      4) write two JSONs:
         - bias_AA_per_residue  → {"C12": {"H": his_bias}, ...}
         - omit_AA_per_residue  → {"C5": ["G"], ...} for mask==True
      5) emit a run.py line binding both JSONs
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    bias_dir = output_dir / "custom_biases"
    omit_dir = output_dir / "custom_omits"
    os.makedirs(bias_dir, exist_ok=True)
    os.makedirs(omit_dir, exist_ok=True)
    os.makedirs(output_dir / "outputs", exist_ok=True)

    script_path = output_dir / "run_ligand_mpnn.sh"
    with open(script_path, "w") as fh:
        fh.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        if gpu_devices:
            fh.write(f"export CUDA_VISIBLE_DEVICES={gpu_devices}\n\n")
        fh.write(
            f'RUN_PY="{run_py}"\n'
            f'MODEL_TYPE="{model_type}"\n'
            f"SEED={seed}\n"
            f"NUM_SEQ={num_seq_per_target}\n"
            f"BATCH={batch_size}\n\n"
        )
        if number_of_batches:
            fh.write(f"NUM_BATCHES={number_of_batches}\n")
        if temperature is not None:
            fh.write(f"TEMP={temperature}\n")
        if checkpoint:
            fh.write(f'CHECKPOINT="{checkpoint}"\n')
        fh.write('GLOBAL_BIAS_AA="Y:1.3,W:1.3,A:-1.3,G:-1.3"\n')
        fh.write('GLOBAL_OMIT_AA="XC"\n')
        fh.write(f'BIAS_DIR="{bias_dir}"\n')
        fh.write(f'OMIT_DIR="{omit_dir}"\n\n')

        pdb_list = list(input_dir.glob("*.pdb"))
        random.shuffle(pdb_list)
        for pdb_path in tqdm(pdb_list):
            base = pdb_path.stem
            outdir = output_dir / "outputs" / base
            fh.write(f'mkdir -p "{outdir}"\n')

            # --- 1) load with ampal, ensure CB ---
            try:
                assembly = load_pdb(str(pdb_path), path=True)
            except Exception as e:
                print(f"[ERROR] cannot load {pdb_path}: {e}")
                continue
            if isinstance(assembly, AmpalContainer):
                assembly = assembly[0]
            # assembly = add_cb_atoms(assembly)

            # build full residue list in mask‐order
            all_residues = assembly.get_monomers(ligands=False)
            res_list = [
                f"{res.parent.id}{res.id}"
                for res in all_residues
            ]

            # --- 2) FE‐proximal HIS bias ---
            nearby = find_residues_near_fe(
                pdb_path, min_dist, max_dist, ca_cb_angle_threshold
            )
            bias_map: Dict[str, Dict[str, float]] = {
                res_id: {"H": his_bias}
                for res_id in nearby
            }
            with open(bias_dir / f"{base}.json", "w") as bf:
                json.dump(bias_map, bf, indent=2)

            # --- 3) parametric mask (must match len(res_list)) ---
            parsed = parse_filename(base.replace("_0", ""))
            n_sel  = int(parsed["residues_per_helix"])
            n_skip = int(parsed["linker_length"])
            n_ribs = int(parsed["orientation_code"][1]) - 1
            param_mask = generate_parametric_residue_mask(
                n_sel, n_skip, n_ribs, diffuse_termini=True
            )
            assert len(param_mask) == len(res_list), (
                f"mask length {len(param_mask)} != #residues {len(res_list)}"
            )

            # --- 4) omit JSON forbidding G where mask==True ---
            omit_map: Dict[str, List[str]] = {
                res_id: ["G"]
                for res_id, flag in zip(res_list, param_mask)
                if flag
            }
            with open(omit_dir / f"{base}.json", "w") as of:
                json.dump(omit_map, of, indent=2)

            # --- 5) write run.py command ---
            cmd = [
                "python $RUN_PY",
                "--model_type $MODEL_TYPE",
                "--seed $SEED",
                f'--pdb_path "{pdb_path}"',
                "--bias_AA $GLOBAL_BIAS_AA",
                "--omit_AA $GLOBAL_OMIT_AA",
                f"--bias_AA_per_residue $BIAS_DIR/{base}.json",
                f"--omit_AA_per_residue $OMIT_DIR/{base}.json",
                f'--out_folder "{outdir}"',
                "--batch_size $BATCH",
            ]
            if number_of_batches:
                cmd.append("--number_of_batches $NUM_BATCHES")
            if temperature is not None:
                cmd.append("--temperature $TEMP")
            if checkpoint:
                cmd.append("--checkpoint_ligand_mpnn $CHECKPOINT")
            cmd += [
                "--pack_side_chains 1",
                "--number_of_packs_per_design 1",
                "--pack_with_ligand_context 1",
            ]

            fh.write(" ".join(cmd) + "\n\n")

    os.chmod(script_path, 0o755)
    return str(script_path)

def heme_covalent_cys(pdb_path: str, threshold: float = 3.5) -> int:
    """
    Load a PDB file and count how many pairs of:
      - ligand atoms named 'CAC' or 'CAB'
      - cysteine SG atoms
    are within `threshold` angstroms of each other.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.
    threshold : float, optional
        Distance cutoff in Å (default 3.5).

    Returns
    -------
    int
        Number of detected contacts.
    """
    # Load structure (Assembly or AmpalContainer)
    structure = ampal.load_pdb(pdb_path, path=True)
    if isinstance(structure, ampal.AmpalContainer):
        # If multiple states, just use the first
        structure = structure[0]

    # Gather all cysteine SG atoms
    cys_sg_atoms = []
    for chain in structure.get_monomers(ligands=False):
        if chain.mol_code == 'CYS':
            sg = chain.atoms.get('SG')
            if sg is not None:
                cys_sg_atoms.append(sg)

    # Gather all ligand atoms named 'CAC' or 'CAB'
    ligand_atoms = []
    for lig in structure.get_ligands(solvent=False):
        for atom in lig.atoms.values():
            if atom.res_label in ('CAC', 'CAB'):
                ligand_atoms.append(atom)

    # Count all SG–ligand pairs within threshold
    contacts = 0
    for sg in cys_sg_atoms:
        sg_coord = sg.array
        for lat in ligand_atoms:
            lat_coord = lat.array
            if np.linalg.norm(sg_coord - lat_coord) <= threshold:
                contacts += 1

    return contacts