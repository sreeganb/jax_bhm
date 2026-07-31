"""
System representation.

Responsibility: turn a declarative description of the system into a built IMP
hierarchy plus its degrees of freedom. Nothing here creates restraints, scores
anything, or samples anything.

Two structural points
---------------------

1. **Copies, not states.** Scaling the system up is done with copies inside a
   single state:

       state.create_molecule("KCOIL", seq, chain_id="A")   -> copy_index 0
       mol.create_copy(chain_id="C")                       -> copy_index 1

   An IMP *state* is an alternative conformation of the same system
   (multi-state modelling); distinct states are not restrained to one another
   or packed together by excluded volume. Copies are genuine extra molecules in
   one assembly, addressable by `copy_index` in IMP.atom.Selection.

2. **Fragments, not one row per molecule.** A PMI topology file describes a
   molecule with as many rows as it has fragments -- e.g. a structured helix,
   an unstructured loop as BEADS, a second structured helix. Each row carries
   its own pdb/range/bead_size/color and its own rigid-body id. A MoleculeSpec
   therefore holds a *list* of FragmentSpec, and rigid bodies are built by
   grouping fragments on their rigid-body id (which may span molecules).
"""

import os
import string
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import IMP
import IMP.atom
import IMP.core
import IMP.pmi.dof
import IMP.pmi.topology


# --------------------------------------------------------------------------
# specifications
# --------------------------------------------------------------------------

@dataclass
class FragmentSpec:
    """
    One row of a topology file: a contiguous stretch of one molecule.

    pdb_file        structure file, or "BEADS"/None for an unstructured stretch
    pdb_chain       chain id to read from `pdb_file`
    residue_range   (first, last) inclusive in sequence numbering, or None for
                    the whole molecule
    pdb_offset      offset applied to PDB numbering
    bead_size       residues per bead for the unstructured case
    color           anything PMI accepts: name, hex string, float, or RGB tuple
    rigid_body      rigid-body id; fragments sharing an id join one rigid body.
                    None means the fragment is sampled as flexible beads.
    resolutions     resolutions built for the structured case
    """
    pdb_file: Optional[str] = None
    pdb_chain: str = "A"
    residue_range: Optional[Tuple[int, int]] = None
    pdb_offset: int = 0
    bead_size: int = 1
    color: Optional[object] = None
    rigid_body: Optional[int] = None
    resolutions: Sequence[int] = (1,)

    def has_structure(self):
        if not self.pdb_file:
            return False
        return str(self.pdb_file).strip().upper() not in ("BEADS", "NONE", "")

    def label(self):
        if self.residue_range:
            span = f"{self.residue_range[0]}-{self.residue_range[1]}"
        else:
            span = "all"
        kind = "struct" if self.has_structure() else "beads"
        rb = f" rb={self.rigid_body}" if self.rigid_body is not None else ""
        return f"[{span} {kind}{rb} color={self.color}]"


@dataclass
class MoleculeSpec:
    """One molecule: a sequence plus the fragments that represent it."""
    name: str
    fasta_file: str
    fasta_id: Optional[str] = None
    fragments: List[FragmentSpec] = field(default_factory=list)

    def sequence_id(self):
        return self.fasta_id if self.fasta_id else self.name

    def effective_fragments(self):
        """
        Fragments to build, defaulting to one whole-molecule beads fragment.

        The default is resolved here rather than in __post_init__ because
        callers legitimately construct a MoleculeSpec with fragments=[] and
        append afterwards; injecting a default at construction time would leave
        a phantom fragment claiming the entire sequence, and the first real
        fragment would then fail with "You have already added representation".
        """
        return self.fragments if self.fragments else [FragmentSpec()]


@dataclass
class SystemSpec:
    """
    The whole system.

    molecules       one MoleculeSpec per distinct molecule
    n_copies        how many copies of the entire molecule set to build
    max_translation / max_rotation
                    rigid-body mover amplitudes
    bead_max_translation
                    flexible-bead mover amplitude
    allow_nonrigid_members
                    if True, unstructured fragments that carry a rigid-body id
                    become non-rigid members of that body. Left False because
                    sampling.imp_blackjax_adapter raises NotImplementedError on
                    non-rigid members: they need a body-frame 3-DOF block that
                    the flat layout [rb_trans | rb_quat | flex_xyz] has no room
                    for. Usable with the replica_exchange sampler only.
    """
    molecules: Sequence[MoleculeSpec]
    n_copies: int = 1
    name: str = "System"
    max_translation: float = 4.0
    max_rotation: float = 0.5
    bead_max_translation: float = 4.0
    allow_nonrigid_members: bool = False

    def __post_init__(self):
        if int(self.n_copies) < 1:
            raise ValueError(f"n_copies must be >= 1, got {self.n_copies}")
        if not self.molecules:
            raise ValueError("SystemSpec requires at least one MoleculeSpec")
        self.n_copies = int(self.n_copies)


@dataclass
class BuiltSystem:
    """Everything downstream code needs about a built system."""
    model: "IMP.Model"
    system: "IMP.pmi.topology.System"
    state: "IMP.pmi.topology.State"
    root_hier: "IMP.atom.Hierarchy"
    dof: "IMP.pmi.dof.DegreesOfFreedom"
    molecules: dict = field(default_factory=dict)   # (name, copy_index) -> Molecule
    spec: Optional[SystemSpec] = None

    @property
    def n_copies(self):
        return self.spec.n_copies if self.spec else 1

    @property
    def molecule_names(self):
        return [m.name for m in self.spec.molecules] if self.spec else []

    def copies_of(self, molecule_name):
        """PMI Molecule objects for every copy of `molecule_name`, in order."""
        return [self.molecules[(molecule_name, i)] for i in range(self.n_copies)]

    def rigid_bodies_and_beads(self):
        """
        Split all leaves into (ordered rigid bodies, flexible/non-rigid beads).

        First-appearance order is preserved so indices stay stable across
        calls -- the JAX adapter relies on this.
        """
        seen = set()
        rigid_bodies = []
        beads = []
        for particle in IMP.atom.get_leaves(self.root_hier):
            rb = None
            if IMP.core.RigidMember.get_is_setup(particle):
                rb = IMP.core.RigidMember(particle).get_rigid_body()
            elif IMP.core.NonRigidMember.get_is_setup(particle):
                rb = IMP.core.NonRigidMember(particle).get_rigid_body()
                beads.append(particle)
            else:
                beads.append(particle)
            if rb is not None and rb not in seen:
                seen.add(rb)
                rigid_bodies.append(rb)
        return rigid_bodies, beads

    def describe(self):
        """Print a compact inventory of what was actually built."""
        rigid_bodies, beads = self.rigid_bodies_and_beads()
        print(f"Built system '{self.spec.name if self.spec else '?'}' "
              f"with {self.n_copies} copy/copies")
        for (name, copy_index), mol in sorted(self.molecules.items()):
            n_leaves = len(IMP.core.get_leaves(mol.get_hierarchy()))
            chain = mol.get_hierarchy().get_name()
            print(f"  {name}.{copy_index:<3d} chain={chain:<4s} beads={n_leaves}")
        print(f"  rigid bodies  : {len(rigid_bodies)}")
        print(f"  flexible beads: {len(beads)}")
        print(f"  movers        : {len(self.dof.get_movers())}")


def copy_index_of(mol):
    """
    Copy index of a PMI Molecule, robust across IMP versions.

    Some releases expose Molecule.get_copy_index(); all of them decorate the
    molecule hierarchy with IMP.atom.Copy, so fall back to that.
    """
    getter = getattr(mol, "get_copy_index", None)
    if callable(getter):
        return int(getter())
    hier = mol.get_hierarchy()
    if IMP.atom.Copy.get_is_setup(hier):
        return int(IMP.atom.Copy(hier).get_copy_index())
    return 0


def _chain_id_alphabet():
    """A, B, ... Z, a, ... z, 0, ... 9, then AA, AB, ... for large systems."""
    singles = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
    for c in singles:
        yield c
    for first in singles:
        for second in singles:
            yield first + second


# --------------------------------------------------------------------------
# topology.txt -> MoleculeSpec
# --------------------------------------------------------------------------

def molecule_specs_from_topology(topology_file, pdb_dir=None, fasta_dir=None,
                                 resolutions=(1,), verbose=True):
    """
    Convert a PMI topology.txt into MoleculeSpec objects.

    Every row becomes a FragmentSpec; rows sharing a molecule name are collected
    into one MoleculeSpec in file order. Colour and rigid-body id are taken
    straight from the parsed component, so they mean exactly what they mean to
    BuildSystem.

    Copy count is controlled by SystemSpec.n_copies, not by repeating rows.
    """
    reader = IMP.pmi.topology.TopologyReader(
        topology_file,
        pdb_dir=pdb_dir if pdb_dir else os.path.dirname(topology_file),
        fasta_dir=fasta_dir if fasta_dir else os.path.dirname(topology_file),
    )

    by_name = {}
    order = []

    for component in reader.get_components():
        name = component.molname
        if name not in by_name:
            by_name[name] = MoleculeSpec(
                name=name,
                fasta_file=component.fasta_file,
                fasta_id=component.fasta_id,
                fragments=[],
            )
            order.append(name)

        residue_range = component.residue_range
        if isinstance(residue_range, str) or residue_range in (None, "all"):
            residue_range = None
        else:
            residue_range = (int(residue_range[0]), int(residue_range[1]))

        # TopologyReader stores rigid_body as a list (usually 0 or 1 entries).
        rb = getattr(component, "rigid_body", None)
        if isinstance(rb, (list, tuple)):
            rb = int(rb[0]) if rb else None
        elif rb in ("", None):
            rb = None
        else:
            rb = int(rb)

        by_name[name].fragments.append(FragmentSpec(
            pdb_file=component.pdb_file,
            pdb_chain=component.chain,
            residue_range=residue_range,
            pdb_offset=int(getattr(component, "pdb_offset", 0) or 0),
            bead_size=int(getattr(component, "bead_size", 1) or 1),
            # Pass the colour through untouched: TopologyReader has already
            # normalised it the same way BuildSystem would consume it.
            color=getattr(component, "color", None),
            rigid_body=rb,
            resolutions=tuple(resolutions),
        ))

    specs = [by_name[n] for n in order]
    if verbose:
        for spec in specs:
            print(f"  [topology] {spec.name}: {len(spec.fragments)} fragment(s)")
            for frag in spec.fragments:
                print(f"      {frag.label()}")
    return specs


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def _residue_selection(mol, residue_range):
    """
    Residues of `mol` covered by `residue_range`, or all of them if None.

    PMI's Molecule.residue_range takes strings for PDB/sequence numbering and
    integers for 0-based indexing, so the bounds are passed as strings.
    """
    if residue_range is None:
        return mol
    first, last = int(residue_range[0]), int(residue_range[1])
    return mol.residue_range(str(first), str(last))


def _add_representation(mol, spec, verbose):
    """
    Attach structure and representation for every fragment of one molecule copy.

    Two passes: all add_structure calls first, then all add_representation
    calls. add_structure only loads coordinates onto residues, while
    add_representation decides how they are beaded, and doing every structure
    load up front means a later fragment cannot see a half-built molecule.

    Returns {rigid_body_id: [residues]} for the fragments that carry one, plus
    the list of residues that should become free flexible beads.
    """
    atomic_by_fragment = {}

    fragments = spec.effective_fragments()

    for index, frag in enumerate(fragments):
        if not frag.has_structure():
            continue
        atomic_by_fragment[index] = mol.add_structure(
            frag.pdb_file,
            chain_id=frag.pdb_chain,
            res_range=frag.residue_range,
            offset=frag.pdb_offset,
            soft_check=True,
        )

    rigid_groups = {}
    free_residues = []

    for index, frag in enumerate(fragments):
        if index in atomic_by_fragment:
            residues = atomic_by_fragment[index]
            mol.add_representation(
                residues,
                resolutions=list(frag.resolutions),
                color=frag.color,
            )
        else:
            residues = _residue_selection(mol, frag.residue_range)
            mol.add_representation(
                residues,
                resolutions=[frag.bead_size],
                color=frag.color,
            )

        if frag.rigid_body is not None:
            rigid_groups.setdefault(frag.rigid_body, []).append(residues)
        else:
            free_residues.append(residues)

        if verbose:
            kind = "structured" if index in atomic_by_fragment else "beads"
            print(f"    {mol.get_name()}.{copy_index_of(mol)} {frag.label()}: {kind}")

    # Any residue not covered by an explicit fragment range still needs a
    # representation, or system.build() will drop it silently.
    leftover = mol.get_non_atomic_residues()
    return rigid_groups, free_residues, leftover


def _setup_degrees_of_freedom(dof, sys_spec, rigid_groups, free_residues,
                              copy_index, verbose):
    """
    Create movers for one copy.

    Fragments are grouped by rigid-body id, so a body may span several
    fragments and several molecules -- which is what a shared id in the
    topology file means. Each copy gets its own bodies.

    Structured fragments with an id -> one rigid body per id, built from the
    structured residues only. Everything else -> independent flexible beads.

    Non-rigid members are deliberately not created: create_rigid_body only
    produces them via its `nonrigid_parts` argument, and the BlackJAX adapter
    rejects them. Passing nonrigid_parts=None keeps every particle either a
    plain rigid member or a free XYZ bead, which is the layout the adapter
    understands.
    """
    for rb_id in sorted(rigid_groups):
        residues = rigid_groups[rb_id]
        if sys_spec.allow_nonrigid_members:
            # Body-frame representation: correct IMP, rejected by the adapter.
            dof.create_rigid_body(
                residues,
                nonrigid_parts=None,
                max_trans=sys_spec.max_translation,
                max_rot=sys_spec.max_rotation,
                nonrigid_max_trans=sys_spec.bead_max_translation,
                name=f"rb{rb_id}_copy{copy_index}",
            )
        else:
            dof.create_rigid_body(
                residues,
                max_trans=sys_spec.max_translation,
                max_rot=sys_spec.max_rotation,
                name=f"rb{rb_id}_copy{copy_index}",
            )
        if verbose:
            print(f"    copy {copy_index}: rigid body {rb_id} "
                  f"from {len(residues)} fragment(s)")

    for residues in free_residues:
        dof.create_flexible_beads(
            residues,
            max_trans=sys_spec.bead_max_translation,
        )
    if verbose and free_residues:
        print(f"    copy {copy_index}: {len(free_residues)} flexible fragment(s)")


def build_system(spec, verbose=True):
    """
    Build the IMP hierarchy and degrees of freedom described by `spec`.

    Returns a BuiltSystem. No restraints are created and nothing is shuffled;
    those are the jobs of restraints.py and samplers.py.
    """
    model = IMP.Model()
    system = IMP.pmi.topology.System(model, name=spec.name)
    state = system.create_state()

    chain_ids = _chain_id_alphabet()
    molecules = {}
    # rigid-body groups and free fragments, accumulated per copy so that a body
    # can span molecules within a copy but never across copies.
    per_copy_rigid = {i: {} for i in range(spec.n_copies)}
    per_copy_free = {i: [] for i in range(spec.n_copies)}

    if verbose:
        print(f"Building '{spec.name}': {len(spec.molecules)} molecule type(s) "
              f"x {spec.n_copies} copy/copies")

    for mol_spec in spec.molecules:
        sequences = IMP.pmi.topology.Sequences(mol_spec.fasta_file)
        sequence = sequences[mol_spec.sequence_id()]

        base = None
        for copy_index in range(spec.n_copies):
            chain_id = next(chain_ids)
            if copy_index == 0:
                mol = state.create_molecule(mol_spec.name, sequence=sequence,
                                            chain_id=chain_id)
                base = mol
            else:
                # create_copy, not create_clone: clones are for
                # symmetry-constrained duplicates and take no independent
                # representation. Copies get identical representation added
                # explicitly and are sampled independently.
                mol = base.create_copy(chain_id=chain_id)

            rigid_groups, free_residues, _ = _add_representation(
                mol, mol_spec, verbose and copy_index == 0
            )

            for rb_id, residues in rigid_groups.items():
                per_copy_rigid[copy_index].setdefault(rb_id, []).extend(residues)
            per_copy_free[copy_index].extend(free_residues)

            molecules[(mol_spec.name, copy_index)] = mol

    root_hier = system.build()

    dof = IMP.pmi.dof.DegreesOfFreedom(model)
    for copy_index in range(spec.n_copies):
        _setup_degrees_of_freedom(
            dof, spec,
            per_copy_rigid[copy_index],
            per_copy_free[copy_index],
            copy_index,
            verbose and copy_index == 0,
        )

    built = BuiltSystem(
        model=model,
        system=system,
        state=state,
        root_hier=root_hier,
        dof=dof,
        molecules=molecules,
        spec=spec,
    )
    if verbose:
        built.describe()
    return built
