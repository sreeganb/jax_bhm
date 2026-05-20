import sys
import IMP
import IMP.core
import IMP.algebra
import IMP.atom
import IMP.container

import IMP.pmi.restraints.crosslinking
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.restraints.em
import IMP.pmi.restraints.basic
import IMP.pmi.tools
import IMP.pmi.samplers
import IMP.pmi.output
import IMP.pmi.macros
import IMP.pmi.topology
import ihm.cross_linkers

import sys
import os

# read in the fasta and pdb 
# Only flexible beads at this point, no rigid bodies 
data_dir = os.path.join(os.getcwd(), "data")
pdb_dir = os.path.join(data_dir, "pdb")
fasta_dir = os.path.join(data_dir, "fasta")

topology_file = os.path.join(data_dir, "topology.txt")

m = IMP.Model()

# Read in topology file
topology = IMP.pmi.topology.TopologyReader(topology_file,
                                           pdb_dir=pdb_dir,
                                           fasta_dir=fasta_dir)

# Sanity check: print parsed components before building
print("Parsed topology components:")
for c in topology.get_components():
    print(f"  mol={c.molname}  pdb={c.pdb_file}  chain={c.chain}  "
          f"res={c.residue_range}  rb={c.rigid_body}")

bs = IMP.pmi.macros.BuildSystem(m, name="ToyModel", resolutions=[1])
bs.add_state(topology)
root_hier, dof = bs.execute_macro()
output = IMP.pmi.output.Output()
output.init_rmf("ini_all.rmf3", [root_hier])
output.write_rmf("ini_all.rmf3")
output.close_rmf("ini_all.rmf3")

output_objects = [] # keep a list of functions that need to be reported
rmf_restraints = []

print("\nBuild complete.")
print("Movers:", dof.get_movers())

# Print molecule/bead counts per type
for state in IMP.atom.get_by_type(root_hier, IMP.atom.STATE_TYPE):
    for mol in IMP.atom.get_by_type(state, IMP.atom.MOLECULE_TYPE):
        leaves = IMP.core.get_leaves(mol)
        print(f"  {mol.get_name()} : {len(leaves)} bead(s)")

molecules = bs.get_molecules()[0]
print("\nMolecules in the system:", molecules)

for mol_names in molecules:
    for mol in molecules[mol_names]:
        cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol)
        cr.set_label(mol.get_name())
        cr.add_to_model()
        print(f"Added connectivity restraint for molecule: {mol.get_name()}")
        print(cr)
        output_objects.append(cr)