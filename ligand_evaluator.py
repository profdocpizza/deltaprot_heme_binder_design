import os
import re
import subprocess
from typing import List, Dict, Optional, Any
import multiprocessing

import ampal
import pandas as pd


def find_pdbs_in_dir(root: str) -> List[str]:
    """Recursively find all .pdb files under root."""
    pdbs = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith('.pdb'):
                pdbs.append(os.path.join(dirpath, fn))
    return pdbs


class MinimizeEvaluator:
    def __init__(
        self,
        pdb_files: List[str],
        ligand_resname: str,
        output_dir: Optional[str] = None,
    ):
        """
        pdb_files: list of paths to input PDBs
        ligand_resname: three-letter code of the ligand (e.g. "LIG")
        output_dir: base folder to hold per-PDB result subfolders; defaults to cwd
        """
        self.pdb_files = pdb_files
        self.ligcode = ligand_resname
        self.base_out = output_dir or os.getcwd()
        os.makedirs(self.base_out, exist_ok=True)

    def _prepare_dirs(self, prefix: str) -> str:
        wd = os.path.join(self.base_out, prefix)
        os.makedirs(wd, exist_ok=True)
        return wd

    def _split(self, pdb: str, workdir: str) -> Dict[str, str]:
        asm = ampal.load_pdb(pdb)
        base = os.path.basename(pdb)
        prefix = base[:-4]

        # write receptor without ligand
        rec_str = asm.make_pdb(ligands=False)
        rec_path = os.path.join(workdir, f"{prefix}_rec.pdb")
        with open(rec_path, "w") as f:
            f.write(rec_str)

        # isolate ligand
        ligs = [m for m in asm.get_ligands(solvent=False) if m.mol_code == self.ligcode]
        if not ligs:
            raise ValueError(f"No ligand {self.ligcode} in {pdb}")
        lig_str = ligs[0].make_pdb()
        lig_path = os.path.join(workdir, f"{prefix}_lig.pdb")
        with open(lig_path, "w") as f:
            f.write(lig_str)

        return {"rec": rec_path, "lig": lig_path}

    def _run_minimize(self, rec: str, lig: str, prefix: str, workdir: str) -> Dict[str, float]:
        """Run gnina minimization and parse affinity & intramolecular energy."""
        out_pdb = os.path.join(workdir, f"{prefix}_min.pdb")
        logf    = os.path.join(workdir, f"{prefix}_minimize.log")

        cmd = [
            "gnina",
            "-r", rec,
            "-l", lig,
            "--autobox_ligand", lig,
            "--minimize",
            "--scoring", "vina",
            "-o", out_pdb,
            "--log", logf
        ]
        subprocess.check_call(cmd, stderr=subprocess.DEVNULL)

        # look for line like "Affinity:  -7.45776  -1.26423 (kcal/mol)"
        with open(logf) as f:
            text = f.read()
        m = re.search(
            r"Affinity:\s*([\-\d\.]+)\s+([\-\d\.]+)\s+\(kcal/mol\)",
            text
        )
        if not m:
            raise RuntimeError(f"Could not parse energies in {logf}")
        return {"affinity": float(m.group(1)), "intramolecular_energy": float(m.group(2))}

    def _process_one(self, pdb: str) -> Dict[str, Any]:
        prefix = os.path.splitext(os.path.basename(pdb))[0]
        wd = self._prepare_dirs(prefix)
        parts = self._split(pdb, wd)
        energies = self._run_minimize(parts["rec"], parts["lig"], prefix, wd)
        return {"pdb": pdb, **energies, "rec_file": parts["rec"], "lig_file": parts["lig"], "out_dir": wd}

    def evaluate(self, n_procs: Optional[int] = None) -> pd.DataFrame:
        # check duplicates
        basenames = [os.path.basename(p) for p in self.pdb_files]
        if len(set(basenames)) != len(basenames):
            dup = [b for b in basenames if basenames.count(b) > 1]
            raise ValueError(f"Duplicate basenames: {set(dup)}")

        # parallel processing
        pool = multiprocessing.Pool(processes=n_procs)
        try:
            records = pool.map(self._process_one, self.pdb_files)
        finally:
            pool.close()
            pool.join()

        return pd.DataFrame(records)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Minimize ligands in PDB(s) with GNINA"
    )
    parser.add_argument(
        "-p", "--pdb_dir", help="Recursively scan this directory for PDB files"
    )
    parser.add_argument(
        "pdbs", nargs="*", help="Explicit PDB file paths"
    )
    parser.add_argument(
        "-l", "--ligcode", required=True, help="Three-letter ligand code"
    )
    parser.add_argument(
        "-d", "--output_dir", help="Base folder for results"
    )
    parser.add_argument(
        "-o", "--out_csv", default="gnina_minimize_summary.csv",
        help="CSV summary filename"
    )
    parser.add_argument(
        "-n", "--n_procs", type=int, default=None,
        help="Number of parallel processes (default: all available CPUs)"
    )
    args = parser.parse_args()

    # collect PDBs
    to_process = []
    if args.pdb_dir:
        to_process.extend(find_pdbs_in_dir(args.pdb_dir))
    if args.pdbs:
        to_process.extend(args.pdbs)
    if not to_process:
        parser.error("Provide --pdb_dir or PDB file paths")
    to_process = list(dict.fromkeys(to_process))

    # run in parallel
    me = MinimizeEvaluator(
        pdb_files=to_process,
        ligand_resname=args.ligcode,
        output_dir=args.output_dir
    )
    df = me.evaluate(n_procs=args.n_procs)
    csv_path = os.path.join(me.base_out, args.out_csv)
    df.to_csv(csv_path, index=False)
    print(f"Done! Summary: {csv_path}")
    print(f"Results folders under: {me.base_out}")
