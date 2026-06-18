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
        
# Add in some JAX based distance restraints between some specific residues
# Apparently the PMI based distance restraint has a JAX implementation,
# so lets test this
tuple_1 = (1, 1, "KCOIL")
tuple_2 = (1, 1, "ECOIL")
distance_max = 10.0  # example distance
kappa = 40.0
ts1 = IMP.core.HarmonicUpperBound(distance_max, kappa)

sel1 = IMP.atom.Selection(root_hier, resolution=1, molecule=tuple_1[2],
                          residue_index=tuple_1[0], copy_index = 0)

particle_1 = sel1.get_selected_particles()
sel2 = IMP.atom.Selection(root_hier, resolution=1, molecule=tuple_2[2],
                          residue_index=tuple_2[0], copy_index = 0)
particle_2 = sel2.get_selected_particles()
model = root_hier.get_model()

dr1 = IMP.core.DistanceRestraint(model, ts1, particle_1[0], particle_2[0])

# shuffle configurations
IMP.pmi.tools.shuffle_configuration(root_hier,
                                    max_translation=500,
                                    avoidcollision_rb=False,
                                    bounding_box=((-100, -100, 0), (100, 100, 100)))


# Set up RMF output
output = IMP.pmi.output.Output()
output.init_rmf("shuffled_particles.rmf3", [root_hier])  # Initialize the RMF file
output.write_rmf("shuffled_particles.rmf3")  # Write the RMF file

sf_imp = IMP.core.RestraintsScoringFunction(dr1)
score = sf_imp.evaluate(False)

ji = dr1._get_jax()
print(ji)
jax_score_func = ji.score_func 
jmodel = ji.get_jax_model()
print("Initial JAX score:", jax_score_func(jmodel))
print("Initial IMP score:", score)
print("jax model:", jmodel)

# write JAX model to file for inspection as a JAX numpy array as a csv file
import numpy as np


def decorator_is_setup(decorator, model, particle_index, particle):
    for args in ((model, particle_index), (particle,), (particle_index,)):
        try:
            return bool(decorator.get_is_setup(*args))
        except TypeError:
            continue
        except Exception:
            return False
    return False


def normalize_particle_index(particle_index):
    if hasattr(particle_index, "get_index"):
        slot_id = int(particle_index.get_index())
    else:
        slot_id = int(particle_index)

    try:
        normalized = IMP.ParticleIndex(slot_id)
    except Exception:
        normalized = slot_id

    return normalized, slot_id


def particle_flags(model, particle_index):
    particle_index, _ = normalize_particle_index(particle_index)
    particle = model.get_particle(particle_index)
    checks = [
        ("Hierarchy", IMP.atom.Hierarchy),
        ("Atom", IMP.atom.Atom),
        ("Residue", IMP.atom.Residue),
        ("Chain", IMP.atom.Chain),
        ("Molecule", IMP.atom.Molecule),
        ("XYZR", IMP.core.XYZR),
        ("RigidBody", IMP.core.RigidBody),
        ("RigidMember", IMP.core.RigidMember),
    ]
    return [
        label for label, decorator in checks
        if decorator_is_setup(decorator, model, particle_index, particle)
    ]


def print_jax_model_diagnostics(model, root_hier, jmodel):
    jm_xyz = np.asarray(jmodel["xyz"])
    jm_r = np.asarray(jmodel["r"])

    xyz_finite = np.isfinite(jm_xyz).all(axis=1)
    r_finite = np.isfinite(jm_r)
    finite_rows = xyz_finite & r_finite
    invalid_rows = ~finite_rows

    particle_indexes = list(model.get_particle_indexes())
    normalized_particle_indexes = [normalize_particle_index(pi) for pi in particle_indexes]
    particle_slot_ids = sorted(slot_id for _, slot_id in normalized_particle_indexes)
    particle_slot_set = set(particle_slot_ids)
    xyzr_particle_count = sum(
        decorator_is_setup(IMP.core.XYZR, model, pi, model.get_particle(pi))
        for pi, _ in normalized_particle_indexes
    )

    leaves = IMP.core.get_leaves(root_hier)
    leaf_particles = [leaf.get_particle() for leaf in leaves]
    leaf_xyzr_count = sum(
        decorator_is_setup(
            IMP.core.XYZR,
            model,
            particle.get_index(),
            particle,
        )
        for particle in leaf_particles
    )
    atom_leaf_count = sum(
        decorator_is_setup(
            IMP.atom.Atom,
            model,
            particle.get_index(),
            particle,
        )
        for particle in leaf_particles
    )

    invalid_slot_ids = np.flatnonzero(invalid_rows)
    invalid_existing_particles = sum(int(i) in particle_slot_set for i in invalid_slot_ids)
    invalid_missing_particles = len(invalid_slot_ids) - invalid_existing_particles

    print("\nJAX model array semantics (from IMP internals):")
    print("  - get_jax_model() exports Model.get_spheres_numpy() as jmodel['xyz'] and jmodel['r']")
    print("  - each row is one slot in IMP's internal sphere table, indexed by particle index")
    print("  - particles without a valid XYZR sphere are stored as (+inf, +inf, +inf; +inf)")

    print("\nJAX model array dimensions:")
    print(f"  xyz shape: {jm_xyz.shape}")
    print(f"  r shape:   {jm_r.shape}")
    print(f"  total sphere-table slots: {len(jm_r)}")
    print(f"  finite XYZR rows:         {finite_rows.sum()}")
    print(f"  inf placeholder rows:     {invalid_rows.sum()}")

    print("\nIMP model bookkeeping:")
    print(f"  model particle count:     {len(particle_indexes)}")
    print(f"  max particle index:       {particle_slot_ids[-1] if particle_slot_ids else -1}")
    print(f"  XYZR-decorated particles: {xyzr_particle_count}")
    print(f"  hierarchy leaf count:     {len(leaves)}")
    print(f"  leaf particles with XYZR: {leaf_xyzr_count}")
    print(f"  atomistic leaves:         {atom_leaf_count}")
    print(f"  non-atom leaves:          {len(leaves) - atom_leaf_count}")

    print("\nHow the inf rows relate to the IMP model:")
    print(f"  invalid slots that still map to an IMP particle: {invalid_existing_particles}")
    print(f"  invalid slots beyond the live particle set:      {invalid_missing_particles}")
    print("  interpretation: these rows are bookkeeping particles or particles without XYZR setup")

    if len(invalid_slot_ids) > 0:
        print("\nFirst invalid sphere-table rows:")
        for row_id in invalid_slot_ids[:20]:
            row_id = int(row_id)
            if row_id in particle_slot_set:
                particle_index = IMP.ParticleIndex(row_id)
                particle = model.get_particle(particle_index)
                flags = particle_flags(model, particle_index)
                flag_text = ", ".join(flags) if flags else "none"
                print(
                    f"  slot {row_id:4d}: particle='{particle.get_name()}' "
                    f"decorators=[{flag_text}]"
                )
            else:
                print(f"  slot {row_id:4d}: no live IMP particle at this index")

    return jm_xyz, jm_r, xyz_finite, r_finite, finite_rows

jm_xyz, jm_r, xyz_finite, r_finite, finite_rows = print_jax_model_diagnostics(
    model, root_hier, jmodel
)

print(
    f"JAX model finite rows: {finite_rows.sum()}/{len(finite_rows)} "
    f"(xyz finite: {xyz_finite.sum()}, r finite: {r_finite.sum()})"
)

# Save full arrays losslessly (keeps inf values, useful for debugging)
np.savez("jax_model.npz", xyz=jm_xyz, r=jm_r)

# Save CSV views for quick inspection
np.savetxt("jax_model_xyz.csv", jm_xyz, delimiter=",")
np.savetxt("jax_model_r.csv", jm_r[:, None], delimiter=",")

# Optional: finite subset only
np.savetxt("jax_model_finite_xyz.csv", jm_xyz[finite_rows], delimiter=",")
np.savetxt("jax_model_finite_r.csv", jm_r[finite_rows, None], delimiter=",")
