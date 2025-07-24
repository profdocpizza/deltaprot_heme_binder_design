import csv
from pathlib import Path
from typing import Union
from ligandmpnn_utils import validate_liganmpnn_heme_binder_sequence

def scan_pdb_dir_with_validate(
    dir_path: Union[str, Path],
    output_csv: Union[str, Path],
    his_fe_cutoff: float = 5.0,
    cationic_heme_cutoff: float = 5.0,
) -> None:
    """
    Recursively scan `dir_path` for .pdb files, assert no duplicate basenames,
    run validate_liganmpnn_heme_binder_sequence (which checks axial His and
    positive-residue contacts), then write CSV:

      file_path,heme_binding_valid

    where the bool is True iff all checks in validate_… passed.
    """
    dir_path = Path(dir_path)
    output_csv = Path(output_csv)

    # gather and sanity-check
    pdbs = list(dir_path.rglob("*.pdb"))
    if not pdbs:
        raise ValueError(f"No .pdb files found under {dir_path!r}")
    stems = [p.stem for p in pdbs]
    dupes = {s for s in stems if stems.count(s) > 1}
    assert not dupes, f"Duplicate PDB basenames found: {dupes}"

    # run validation
    rows = []
    for pdb in pdbs:
        ok = validate_liganmpnn_heme_binder_sequence(
            pdb,
            his_fe_cutoff=his_fe_cutoff,
            cationic_heme_anionic_cutoff=cationic_heme_cutoff,
        )
        rows.append((str(pdb), ok))

    # write CSV
    with output_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_path", "heme_binding_residues_present"])
        writer.writerows(rows)


scan_pdb_dir_with_validate(
    "/home/tadas/code/deltaprot_heme_binder_design/outputs/5_structure_prediction/outputs_HEM",
    "/home/tadas/code/deltaprot_heme_binder_design/outputs/6_evaluation/heme_binding_residues_check.csv",
)
