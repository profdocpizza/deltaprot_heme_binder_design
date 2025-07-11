import ampal
import numpy as np
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