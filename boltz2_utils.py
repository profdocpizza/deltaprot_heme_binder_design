import os
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
import yaml
import json
from typing import Dict, Any
import pandas as pd

def sanitize_chain_id(raw_id: str) -> str:
    """Remove underscores for compatibility with Boltz's chain name parsing."""
    return raw_id.replace('_', '')

# def make_boltz2_cofold_script(input_fasta, fold_outputs_base_dir):
#     """Generates input YAML files and a run script to co-fold protein sequences with heme using Boltz-2."""
#     base = Path(fold_outputs_base_dir)
#     inputs_dir = base / "inputs"
#     outputs_dir = base / "outputs"

#     # Create directories
#     inputs_dir.mkdir(parents=True, exist_ok=True)

#     # Parse the input FASTA and write per-chain YAML
#     for record in SeqIO.parse(input_fasta, "fasta"):
#         yaml_file = inputs_dir / f"{record.id}.yaml"

#         data = {
#             'version': 1,
#             'sequences': [
#                 {
#                     'protein': {
#                         'id': "A",
#                         'sequence': str(record.seq),
#                         'msa': "empty"
#                     }
#                 },
#                 {
#                     'ligand': {
#                         'id': "B",
#                         'ccd': 'HEM'
#                     }
#                 }
#             ],
#             'properties': [
#                 {
#                     'affinity': {
#                         'binder': 'B'
#                     }
#                 }
#             ]
#         }
#         with open(yaml_file, 'w') as f:
#             yaml.dump(data, f)

#     # Generate the shell script to run all YAMLs in inputs/
#     script_file = base / "run_boltz2_cofold.sh"
#     with open(script_file, "w") as f:
#         f.write("#!/usr/bin/env bash\n")
#         f.write("set -euo pipefail\n\n")
#         f.write("# Navigate to script directory\n")
#         f.write("cd \"$(dirname \"${BASH_SOURCE[0]}\")\"\n\n")
#         f.write("# Ensure outputs directory exists\n")
#         f.write(f"mkdir -p {outputs_dir}\n\n")
#         f.write("# Run Boltz-2 predictions on all YAMLs in inputs/ directory\n")
#         f.write(f"boltz predict inputs/ --use_msa_server --preprocessing-threads 1 --out_dir {outputs_dir}\n")

#     # Make the script executable
#     os.chmod(script_file, 0o755)


def make_boltz2_cofold_script(input_fasta, fold_outputs_base_dir, fold_with_ligand=None):
    """
    Generates input YAML files and a run script to fold protein sequences, optionally co-folding with a ligand using Boltz-2.

    Parameters:
    - input_fasta (str or Path): Path to the FASTA file containing protein sequences.
    - fold_outputs_base_dir (str or Path): Base directory for inputs and outputs.
    - fold_with_ligand (str, optional): CCD identifier for the ligand (e.g., 'HEM').
      If None, generates inputs for protein-only folding.
    """
    base = Path(fold_outputs_base_dir)
    # Define input and output directories based on ligand option
    if fold_with_ligand:
        inputs_dir = base / f"inputs_{fold_with_ligand}"
        outputs_dir = base / f"outputs_{fold_with_ligand}"
    else:
        inputs_dir = base / "inputs_no_lig"
        outputs_dir = base / "outputs_no_lig"

    # Create directories
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Parse the input FASTA and write per-chain YAML
    for record in SeqIO.parse(input_fasta, "fasta"):
        yaml_file = inputs_dir / f"{record.id}.yaml"
        # Base YAML structure with protein sequence
        data = {
            'version': 1,
            'sequences': [
                {
                    'protein': {
                        'id': 'A',
                        'sequence': str(record.seq),
                        'msa': 'empty'
                    }
                }
            ]
        }
        # If ligand requested, append ligand and affinity properties
        if fold_with_ligand:
            data['sequences'].append({
                'ligand': {
                    'id': 'B',
                    'ccd': fold_with_ligand
                }
            })
            data['properties'] = [
                {
                    'affinity': {
                        'binder': 'B'
                    }
                }
            ]

        # Write YAML file
        with open(yaml_file, 'w') as f:
            yaml.dump(data, f)

    # Generate the shell script to run Boltz-2 predictions
    script_file = base / f"run_boltz2_{fold_with_ligand or 'no_lig'}.sh"
    with open(script_file, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write("# Navigate to script directory\n")
        f.write("cd \"$(dirname \"${BASH_SOURCE[0]}\")\"\n\n")
        f.write("# Ensure outputs directory exists\n")
        f.write(f"mkdir -p {outputs_dir}\n\n")
        f.write("# Run Boltz-2 predictions on all YAMLs in inputs directory\n")
        f.write(f"boltz predict {inputs_dir} --output_format pdb --preprocessing-threads 25 --out_dir {outputs_dir}\n")

    # Make the script executable
    os.chmod(script_file, 0o755)

    print(f"Generated inputs in {inputs_dir} and script {script_file}")


# def parse_boltz2_dir(main_dir: str) -> pd.DataFrame:
#     """
#     Parse a boltz2-style directory into a flat pandas DataFrame.

#     Args:
#         main_dir: path to the main directory which contains a 'predictions' subdirectory.

#     Returns:
#         DataFrame with columns:
#           - sequence_name
#           - path_to_pdb
#           - confidence_score, ptm, iptm, ...
#           - chains_ptm_{chain}
#           - pair_chains_iptm_{chain0}_{chain1}
#     """
#     main_path = Path(main_dir)
#     preds_dir = main_path / "predictions"
#     records = []

#     print(f"parsing {len(list(preds_dir.iterdir()))} predictions")
#     for seq_dir in tqdm(preds_dir.iterdir()):
#         if not seq_dir.is_dir():
#             print(f"skipping non-directory {seq_dir}. This is unexpected")
#             continue

#         affinity_json_path = seq_dir / f"affinity_{seq_dir.name}.json"
#         confidence_json_path = seq_dir / f"confidence_{seq_dir.name}_model_0.json"
#         pdb_path = seq_dir / f"{seq_dir.name}_model_0.pdb"

#         # create initial dict
#         orientation_code = seq_dir.name.split("_yaw")[0]

#         flat: Dict[str, Any] = {
#             "orientation_code": orientation_code,
#             "sequence_name": seq_dir.name,
#             "path_to_pdb": str(pdb_path.resolve())
#         }

#         # add confidence json keys
#         with confidence_json_path.open() as f:
#             confidence_data = json.load(f)
        
#         for key, val in confidence_data.items():
#             if key in ("chains_ptm", "pair_chains_iptm"):
#                 continue
#             flat[key] = val

#         for chain, score in confidence_data.get("chains_ptm", {}).items():
#             flat[f"chains_ptm_{chain}"] = score

#         for c0, inner in confidence_data.get("pair_chains_iptm", {}).items():
#             for c1, score in inner.items():
#                 flat[f"pair_chains_iptm_{c0}_{c1}"] = score

#         records.append(flat)

#         # add affinity json keys
#         with affinity_json_path.open() as f:
#             affinity_data = json.load(f)

#         for key, val in affinity_data.items():
#             flat[key] = val

#     df = pd.DataFrame(records)
#     # optional: sort columns so sequence_name, path_to_pdb come first
#     cols = ["sequence_name","orientation_code", "path_to_pdb"] + [c for c in df.columns if c not in ("sequence_name","orientation_code", "path_to_pdb")]
#     return df[cols]

def parse_boltz2_dir(main_dir: str, expect_ligand: bool = True) -> pd.DataFrame:
    """
    Parse a boltz2-style directory into a flat pandas DataFrame.

    Args:
        main_dir: path to the main directory which contains a 'predictions' subdirectory.
        expect_ligand: if False, skip affinity data and prefix all other columns with 'no_lig_'.

    Returns:
        DataFrame with columns:
          - sequence_name
          - path_to_pdb
          - confidence_score, ptm, iptm, ...
          - chains_ptm_{chain}
          - pair_chains_iptm_{chain0}_{chain1}
          - affinity-related if expect_ligand=True
    """
    main_path = Path(main_dir)
    preds_dir = main_path / "predictions"
    records = []

    print(f"parsing {len(list(preds_dir.iterdir()))} predictions")
    for seq_dir in tqdm(preds_dir.iterdir()):
        if not seq_dir.is_dir():
            print(f"skipping non-directory {seq_dir}. This is unexpected")
            continue

        affinity_json_path = seq_dir / f"affinity_{seq_dir.name}.json"
        confidence_json_path = seq_dir / f"confidence_{seq_dir.name}_model_0.json"
        pdb_path = seq_dir / f"{seq_dir.name}_model_0.pdb"

        # create initial dict
        flat: dict = {
            "sequence_name": seq_dir.name,
            "path_to_pdb": str(pdb_path.resolve())
        }

        # add confidence json keys
        with confidence_json_path.open() as f:
            confidence_data = json.load(f)
        for key, val in confidence_data.items():
            if key in ("chains_ptm", "pair_chains_iptm"):
                continue
            flat[key] = val
        for chain, score in confidence_data.get("chains_ptm", {}).items():
            flat[f"chains_ptm_{chain}"] = score
        for c0, inner in confidence_data.get("pair_chains_iptm", {}).items():
            for c1, score in inner.items():
                flat[f"pair_chains_iptm_{c0}_{c1}"] = score

        # add affinity json keys only if expected
        if expect_ligand:
            with affinity_json_path.open() as f:
                affinity_data = json.load(f)
            for key, val in affinity_data.items():
                flat[key] = val

        records.append(flat)

    df = pd.DataFrame(records)
    # order columns
    cols = ["sequence_name", "path_to_pdb"] + [c for c in df.columns if c not in ("sequence_name", "path_to_pdb")]
    df = df[cols]

    # if no ligand, prefix remaining columns
    if not expect_ligand:
        rename_map = {c: f"no_lig_{c}" for c in df.columns if c not in ("sequence_name")}
        df = df.rename(columns=rename_map)

    return df