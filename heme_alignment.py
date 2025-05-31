import ampal
from pipeline_utils import pipeline_data
import numpy as np
from dp_utils.helix_assembly.utils import apply_transform_to_assembly


def get_heme_transform(assembly,
                       radius_atoms=("CHC","CHA"),
                       plane_atoms=("NA","NB","NC"),
                       center_atom="FE"):
    # --- gather coords ---
    coords = { atom.res_label: atom._vector
               for atom in assembly.get_atoms() }
    a, b = coords[radius_atoms[0]], coords[radius_atoms[1]]
    p1, p2, p3 = (coords[x] for x in plane_atoms[:3])
    c = coords[center_atom]

    # --- Rodrigues’ helper ---
    def rodrigues(v1, v2):
        v1 = v1/np.linalg.norm(v1)
        v2 = v2/np.linalg.norm(v2)
        axis = np.cross(v1, v2)
        if np.linalg.norm(axis) < 1e-8:
            return np.eye(3)
        axis /= np.linalg.norm(axis)
        θ = np.arccos(np.clip(v1.dot(v2), -1, 1))
        K = np.array([[    0, -axis[2],  axis[1]],
                      [ axis[2],     0, -axis[0]],
                      [-axis[1], axis[0],      0]])
        return np.eye(3) + np.sin(θ)*K + (1 - np.cos(θ))*(K @ K)

    # --- 1) align CHC→CHA to +Y ---
    R1 = rodrigues(b - a, np.array([0,1,0]))

    # --- 2) compute ring‐normal, bring it to +Z by yaw around Y ---
    normal = np.cross(p2 - p1, p3 - p1)
    normal = R1 @ normal
    normal /= np.linalg.norm(normal)
    proj = normal.copy(); proj[1] = 0
    proj /= np.linalg.norm(proj)
    φ = np.arctan2(-proj[0], proj[2])
    R2 = np.array([[ np.cos(φ), 0, np.sin(φ)],
                   [         0, 1,         0],
                   [-np.sin(φ), 0, np.cos(φ)]])
    R = R2 @ R1

    # --- 3) translations to zero out FE ---
    t_post = - R @ c    # if rotate first
    t_pre  = - c        # if translate first

    return R, t_post, t_pre


def align_and_save_heme_pdb():
    ligand_assembly = ampal.load_pdb(pipeline_data["ligand"]["raw_file_path"])
    # apply it:
    R, t_post, t_pre = get_heme_transform(ligand_assembly,
                            radius_atoms=["CHC","CHA"],
                            plane_atoms=["C3D", "C2D", "C1D", "CHD", "C4C", "C3C", "C2C", "C1C", "CHC", "C4B", "C3B", "C2B", "C1B", "CHB", "C4A", "C3A", "C2A", "C1A", "CHA", "C4D"], #,["NA","NB","NC","ND"],
                            center_atom="FE")

    apply_transform_to_assembly(ligand_assembly,
                                rotation_matrix=R,
                                translation_vector=t_pre,
                                translate_first=True)

    # save ligand_assembly.pdb
    file_path  = pipeline_data["ligand"]["file_path"]
    with open(file_path, "w") as f:
        f.write(ligand_assembly.pdb)
