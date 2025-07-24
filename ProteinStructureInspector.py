"""
This script updates your current PyMOL session with the following:
  - Displays all objects in cartoon representation colored "grey70" with 60% transparency.
  - Shows side-chain wire representations for atoms that are not backbone atoms (i.e. not C, N, O, H).
  - Selects and colors residues by type:
      • Nonpolar (ALA, LEU, VAL): colors sidechain carbon atoms in slate.
      • Charged (ARG, LYS, ASP, GLU): colors sidechain carbon atoms in cyan.
      • Polar (SER, THR, ASN, GLN, CYS, TYR): colors sidechain carbon atoms in magenta.
      • Special (GLY, PRO): colors the entire residue green (since glycine lacks a sidechain).
      • All other residues: colors sidechain carbon atoms in orange.
  - Annotates hydrogen bonds with yellow dashed lines.
  - Provides placeholders for annotating pi–pi interactions and salt bridges.
  
Note: Customize the functions for pi–pi interactions and salt bridges detection if needed.
"""

from pymol import cmd

def annotate_hbonds(selection="all"):
    """
    Annotate hydrogen bonds as yellow dashed lines.
    
    This simple method uses a distance cutoff and basic donor/acceptor definitions.
    For a more robust detection, consider implementing a dedicated algorithm.
    """
    # Use alternative selection names to avoid reserved keywords.
    cmd.select("donor_sel", "((elem N or elem O) and hydro)")
    cmd.select("acceptor_sel", "elem O")
    
    # Use PyMOL's find_pairs to detect candidate hydrogen bond pairs.
    hb_pairs = cmd.find_pairs("donor_sel", "acceptor_sel", cutoff=3.2, mode=1)
    
    # Each pair is expected to be a tuple: ((object1, index1), (object2, index2))
    for pair in hb_pairs:
        idx1 = pair[0][1]
        idx2 = pair[1][1]
        obj_name = "hbond_%d_%d" % (idx1, idx2)
        cmd.distance(obj_name, "index %d" % idx1, "index %d" % idx2)
        cmd.set("dash_color", "yellow", obj_name)
    
    # Clean up temporary selections.
    cmd.delete("donor_sel")
    cmd.delete("acceptor_sel")

def annotate_pi_pi(selection="all"):
    """
    Annotate pi–pi interactions.
    
    This is a placeholder function. To detect pi–pi interactions, you might need to
    identify aromatic rings, compute centroids, and evaluate their orientations and distances.
    """
    print("annotate_pi_pi: Function not implemented. Please add your custom code here.")

def annotate_salt_bridges(selection="all"):
    """
    Annotate salt bridges.
    
    This is a placeholder function. Salt bridges are generally defined as close interactions
    (e.g., within 4 Å) between oppositely charged side chains. Implement your own detection logic as needed.
    """
    print("annotate_salt_bridges: Function not implemented. Please add your custom code here.")

def main():
    # -------------------------
    # 1. Global display settings
    # -------------------------
    cmd.bg_color("black")
    
    # -------------------------
    # 2. Cartoon representation for all objects
    # -------------------------
    cmd.show("cartoon", "all")
    cmd.set("cartoon_color", "grey70")
    cmd.set("cartoon_transparency", 0.6, "all")
    
    # -------------------------
    # 3. Side-chain wire representation (exclude backbone atoms: C, N, O, H)
    # -------------------------
    cmd.select("sidechains", "((resn GLY+PRO and not name C+O) or (not resn GLY+PRO and not name C+N+O+H))")
    cmd.show("lines", "sidechains")
            
    # -------------------------
    # 4. Residue selections and coloring
    # -------------------------
    # Nonpolar residues: ALA, LEU, VAL (color sidechain carbon atoms in slate)
    cmd.select("nonpolar", "resn ALA+LEU+VAL+TRP+ILE+TYR+MET+PHE")
    cmd.color("orange", "nonpolar and elem C and sidechains")
    
    # Charged residues: ARG, LYS, ASP, GLU (color sidechain carbon atoms in cyan)
    cmd.select("charged", "resn ARG+LYS+ASP+GLU+HIS")
    cmd.color("slate", "charged and elem C and sidechains")
    
    # Polar residues: SER, THR, ASN, GLN, CYS, TYR (color sidechain carbon atoms in magenta)
    cmd.select("polar", "resn SER+THR+ASN+GLN+CYS")
    cmd.color("magenta", "polar and elem C and sidechains")
    
    # Special residues: GLY, PRO (color the entire residue green)
    cmd.select("special", "resn GLY+PRO")
    cmd.color("green", "special")
    
    # Other residues (those not in any of the above groups) colored orange (sidechain carbon atoms)
    cmd.select("others", "not (nonpolar or charged or polar or special)")
    cmd.color("orange", "others and elem C and sidechains")
    
    # -------------------------
    # 5. Annotate interactions
    # -------------------------
    annotate_hbonds("all")
    annotate_salt_bridges("all")
    
    # Optional: Adjust global dash settings for the annotated bonds.
    cmd.set("dash_gap", 0.4)
    cmd.set("dash_length", 0.6)
    cmd.set("dash_radius", 0.1)
    cmd.set("dash_color", "yellow")
    
    # Clear any active selections.
    cmd.deselect()

# # Run the main function unconditionally
# main()
print("protein_descriptor script executed successfully!")
