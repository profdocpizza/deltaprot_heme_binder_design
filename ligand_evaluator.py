import os
import re
import subprocess
from typing import List, Dict, Optional

import ampal
import pandas as pd


class DockEvaluator:
    def __init__(
        self,
        pdb_files: List[str],
        ligand_resname: str,
        flex_dist: float = 3.5,
        flex_max: int = 5,
        output_dir: Optional[str] = None,
    ):
        """
        pdb_files: list of paths to input PDBs
        ligand_resname: three-letter code of the ligand to isolate (e.g. "LIG")
        flex_dist: radius (Å) around ligand within which to make side chains flexible
        flex_max: maximum number of flexible residues to include
        output_dir: base folder to hold per-PDB result subfolders; defaults to cwd
        """
        self.pdb_files = pdb_files
        self.ligcode = ligand_resname
        self.flex_dist = flex_dist
        self.flex_max = flex_max
        # set base output directory
        self.base_out = output_dir or os.getcwd()
        os.makedirs(self.base_out, exist_ok=True)

    def _prepare_dirs(self, prefix: str) -> str:
        """Create and return a working directory for a single PDB prefix."""
        wd = os.path.join(self.base_out, prefix)
        os.makedirs(wd, exist_ok=True)
        return wd

    def _split(self, pdb: str, workdir: str) -> Dict[str, str]:
        """Use AMPAL to write receptor- and ligand-only PDBs into workdir. Returns dict with paths."""
        asm = ampal.load_pdb(pdb)
        base = os.path.basename(pdb)
        prefix = base[:-4] if base.lower().endswith('.pdb') else base

        # receptor: all polymers, no ligands
        rec_str = asm.make_pdb(ligands=False)
        rec_path = os.path.join(workdir, f"{prefix}_rec.pdb")
        with open(rec_path, "w") as f:
            f.write(rec_str)

        # ligand: pick all non-solvent ligands matching code
        ligs = [m for m in asm.get_ligands(solvent=False) if m.mol_code == self.ligcode]
        if not ligs:
            raise ValueError(f"No ligand with resname {self.ligcode} found in {pdb}")
        if len(ligs) > 1:
            raise ValueError(f"Multiple ligands with resname {self.ligcode} in {pdb}")
        lig_str = ligs[0].make_pdb()
        lig_path = os.path.join(workdir, f"{prefix}_lig.pdb")
        with open(lig_path, "w") as f:
            f.write(lig_str)

        return {"rec": rec_path, "lig": lig_path}

    def _run_static(self, rec: str, lig: str, prefix: str, workdir: str) -> float:
        """Score-only run, logs to file and returns best Vina affinity."""
        logf = os.path.join(workdir, f"{prefix}_static.log")
        cmd = [
            "gnina", "-r", rec, "-l", lig,
            "--autobox_ligand", lig,
            "--score_only",
            "--cnn_scoring", "none", "--scoring", "vina",
            "--log", logf
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
        # parse from console output if no log created
        scores = [float(x) for x in re.findall(r"(-?\d+\.\d+)", out)]
        if not scores:
            # try parsing log file
            with open(logf) as f:
                text = f.read()
            scores = [float(x) for x in re.findall(r"(-?\d+\.\d+)", text)]
        if not scores:
            raise RuntimeError(f"No static scores parsed for {prefix}")
        return min(scores)

    def _run_flex(self, rec: str, lig: str, prefix: str, workdir: str) -> float:
        """Flexible docking run; logs to file and returns best Vina affinity from the log."""
        out_sdf = os.path.join(workdir, f"{prefix}_flex.sdf.gz")
        logf    = os.path.join(workdir, f"{prefix}_flex.log")
        cmd = [
            "gnina", "-r", rec, "-l", lig,
            "--autobox_ligand", lig,
            "--flexdist_ligand", lig,
            "--flexdist", str(self.flex_dist),
            "--flex_max", str(self.flex_max),
            "--cnn_scoring", "none", "--scoring", "vina",
            "-o", out_sdf,
            "--log", logf,
        ]
        subprocess.check_call(cmd, stderr=subprocess.DEVNULL)

        with open(logf) as f:
            text = f.read()
        matches = re.findall(r"^\s*\d+\s+(-?\d+\.\d+)", text, re.MULTILINE)
        scores = [float(s) for s in matches]
        if not scores:
            raise RuntimeError(f"No flexible-docking scores found in {logf}")
        return min(scores)

    def evaluate(self) -> pd.DataFrame:
        """
        Loop over all PDBs, create per-file subfolder, do split + static + flex,
        return a pandas DataFrame with columns:
        ['pdb', 'static_affinity', 'flex_affinity', 'rec_file', 'lig_file', 'out_dir']
        """
        records = []
        for pdb in self.pdb_files:
            base = os.path.basename(pdb)
            prefix = base[:-4] if base.lower().endswith('.pdb') else base
            wd = self._prepare_dirs(prefix)
            parts = self._split(pdb, wd)
            static_score = self._run_static(parts["rec"], parts["lig"], prefix, wd)
            flex_score   = self._run_flex(parts["rec"], parts["lig"], prefix, wd)
            records.append({
                "pdb": pdb,
                "static_affinity": static_score,
                "flex_affinity": flex_score,
                "rec_file": parts["rec"],
                "lig_file": parts["lig"],
                "out_dir": wd
            })
        return pd.DataFrame(records)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate a batch of PDBs with GNINA (static & side-chain flex docking)"
    )
    parser.add_argument(
        "pdbs", nargs="+",
        help="Input PDB filenames (each must contain exactly one ligand of the specified code)"
    )
    parser.add_argument(
        "-l", "--ligcode", required=True,
        help="Three-letter residue name of the ligand (e.g. LIG)"
    )
    parser.add_argument(
        "--flex_dist", type=float, default=3.5,
        help="Distance cutoff (Å) around ligand for flexible side chains"
    )
    parser.add_argument(
        "--flex_max", type=int, default=5,
        help="Maximum number of flexible residues to include"
    )
    parser.add_argument(
        "-o", "--out_csv", default="gnina_static_flex_affinity.csv",
        help="Write summary table to CSV"
    )
    parser.add_argument(
        "-d", "--output_dir", default=None,
        help="Base folder for per-PDB result subfolders"
    )
    args = parser.parse_args()

    de = DockEvaluator(
        pdb_files=args.pdbs,
        ligand_resname=args.ligcode,
        flex_dist=args.flex_dist,
        flex_max=args.flex_max,
        output_dir=args.output_dir
    )
    df = de.evaluate()
    out_csv = os.path.join(de.base_out, args.out_csv)
    df.to_csv(out_csv, index=False)
    print(f"Done! Summary written to {out_csv}.")
    print(f"Results folders under: {de.base_out}")
