"""
Restraint construction.

Responsibility: given a BuiltSystem, create restraints and hand back a
RestraintBundle containing both the PMI-style wrappers (needed by
ReplicaExchange for stat output) and the raw IMP restraints (needed to build a
ScoringFunction for the JAX samplers).

The three restraint families requested, in the order they are applied:

  1. ConnectivityRestraint  -- keeps each molecule copy contiguous
  2. ExcludedVolumeSphere   -- keeps every bead in the system out of every
                               other bead (a single global restraint, so it
                               covers intra-copy *and* inter-copy overlap)
  3. Harmonic distances     -- arbitrary residue-to-residue restraints declared
                               as DistanceRestraintSpec objects

Distance restraints are declared once and expanded over copies by `scope`,
so scaling `n_copies` from 1 to 100 needs no change to the restraint list.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import IMP
import IMP.atom
import IMP.core
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.tools


# --------------------------------------------------------------------------
# selection helper
# --------------------------------------------------------------------------

def select_one_particle(root_hier, molecule, residue_index, copy_index=0,
                        resolution=1):
    """
    Return exactly one particle, or raise with enough context to debug.

    The previous code silently took particles[0] from a possibly multi-particle
    selection; if a residue maps into a multi-residue bead that is ambiguous, so
    it is worth being explicit about it.
    """
    sel = IMP.atom.Selection(
        root_hier,
        resolution=resolution,
        molecule=str(molecule),
        residue_index=int(residue_index),
        copy_index=int(copy_index),
    )
    particles = sel.get_selected_particles()
    if not particles:
        raise ValueError(
            f"No particle selected for molecule={molecule} "
            f"residue={residue_index} copy={copy_index} resolution={resolution}. "
            "Check the molecule name, that the residue is inside the built "
            "range, and that n_copies is large enough for this copy index."
        )
    return particles[0]


# --------------------------------------------------------------------------
# specification
# --------------------------------------------------------------------------

@dataclass
class DistanceRestraintSpec:
    """
    One harmonic distance restraint, declared once and expanded over copies.

    molecule1/residue1, molecule2/residue2
        the two endpoints
    distance, kappa
        harmonic mean and force constant (IMP.core.Harmonic)
    scope
        "intra_copy"       -> one restraint inside each copy i
        "inter_copy_chain" -> restraint between copy i and copy i+1, for all i
        "inter_copy_ring"  -> as above, plus a closing (N-1, 0) restraint
        "explicit"         -> a single restraint using copy1/copy2 as given
    copy1, copy2
        only used when scope == "explicit"
    label
        optional name used in stat/log output
    """
    molecule1: str
    residue1: int
    molecule2: str
    residue2: int
    distance: float = 10.0
    kappa: float = 1.0
    scope: str = "intra_copy"
    copy1: int = 0
    copy2: int = 0
    label: Optional[str] = None

    VALID_SCOPES = ("intra_copy", "inter_copy_chain", "inter_copy_ring", "explicit")

    def __post_init__(self):
        if self.scope not in self.VALID_SCOPES:
            raise ValueError(
                f"scope must be one of {self.VALID_SCOPES}, got '{self.scope}'"
            )

    def copy_pairs(self, n_copies):
        """Expand this spec into the list of (copy_i, copy_j) it applies to."""
        if self.scope == "intra_copy":
            return [(i, i) for i in range(n_copies)]
        if self.scope in ("inter_copy_chain", "inter_copy_ring"):
            if n_copies < 2:
                return []
            pairs = [(i, i + 1) for i in range(n_copies - 1)]
            if self.scope == "inter_copy_ring" and n_copies > 2:
                pairs.append((n_copies - 1, 0))
            return pairs
        return [(int(self.copy1), int(self.copy2))]

    def default_label(self, copy_i, copy_j):
        if self.label:
            return f"{self.label}_{copy_i}_{copy_j}"
        return (f"{self.molecule1}{self.residue1}.{copy_i}-"
                f"{self.molecule2}{self.residue2}.{copy_j}")


# --------------------------------------------------------------------------
# a PMI-compatible harmonic distance restraint
# --------------------------------------------------------------------------

class HarmonicDistanceRestraint:
    """
    Thin PMI-style wrapper around IMP.core.DistanceRestraint + IMP.core.Harmonic.

    IMP.pmi.restraints.basic.DistanceRestraint uses a flat-bottomed well
    (harmonic lower + upper bound). When a genuine single-minimum harmonic is
    wanted, this wrapper gives it while still exposing the PMI interface
    (add_to_model / get_restraint / get_output / set_weight) that
    ReplicaExchange expects from an output object.
    """

    def __init__(self, model, particle_pairs, distance, kappa, label="Harmonic",
                 weight=1.0):
        self.model = model
        self.label = str(label)
        self.weight = float(weight)
        self.restraint_set = IMP.RestraintSet(model, f"HarmonicDistance_{self.label}")
        self._pairs = []

        score = IMP.core.Harmonic(float(distance), float(kappa))
        for p1, p2 in particle_pairs:
            restraint = IMP.core.DistanceRestraint(model, score, p1, p2)
            self.restraint_set.add_restraint(restraint)
            self._pairs.append((p1, p2))

        self.restraint_set.set_weight(self.weight)

    def add_to_model(self):
        IMP.pmi.tools.add_restraint_to_model(self.model, self.restraint_set)
        return self

    def get_restraint(self):
        return self.restraint_set

    def set_weight(self, weight):
        self.weight = float(weight)
        self.restraint_set.set_weight(self.weight)

    def evaluate(self):
        return float(self.restraint_set.unprotected_evaluate(None)) * self.weight

    def get_output(self):
        """Stat-file entries, matching the PMI restraint output convention."""
        output = {"_TotalScore": str(self.evaluate())}
        output[f"HarmonicDistanceRestraint_{self.label}"] = str(self.evaluate())
        for i, (p1, p2) in enumerate(self._pairs):
            d = IMP.core.get_distance(IMP.core.XYZ(p1), IMP.core.XYZ(p2))
            output[f"HarmonicDistanceRestraint_{self.label}_d{i}"] = str(d)
        return output


# --------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------

@dataclass
class RestraintBundle:
    """
    The product of RestraintBuilder.

    wrappers          PMI-style objects (used as ReplicaExchange output_objects)
    restraints        raw IMP restraints/restraint sets (used for ScoringFunction)
    scoring_function  IMP.core.RestraintsScoringFunction over `restraints`
    summary           counts per restraint family, for logging
    """
    wrappers: List[object] = field(default_factory=list)
    restraints: List[object] = field(default_factory=list)
    scoring_function: Optional[object] = None
    summary: dict = field(default_factory=dict)

    def evaluate(self):
        return float(self.scoring_function.evaluate(False))

    def describe(self):
        print("Restraints")
        for key in sorted(self.summary):
            print(f"  {key}: {self.summary[key]}")
        print(f"  scoring function terms: {len(self.restraints)}")
        print(f"  current score: {self.evaluate():.4f}")


# --------------------------------------------------------------------------
# builder
# --------------------------------------------------------------------------

class RestraintBuilder:
    """
    Fluent builder so a user script reads as a declaration of intent:

        bundle = (RestraintBuilder(built)
                  .connectivity()
                  .excluded_volume(resolution=1)
                  .distances(DISTANCE_RESTRAINTS)
                  .finalize())

    Every method adds its restraints to the IMP model immediately (so that
    ReplicaExchange sees them) and records them for the ScoringFunction that
    finalize() builds (so that the JAX samplers see exactly the same terms).
    """

    def __init__(self, built, verbose=True):
        self.built = built
        self.verbose = verbose
        self._wrappers = []
        self._restraints = []
        self._summary = {}

    # -- internal -----------------------------------------------------------

    def _register(self, wrapper, family):
        """Add a PMI-style wrapper to the model and record its IMP restraint."""
        wrapper.add_to_model()
        self._wrappers.append(wrapper)

        for getter in ("get_restraint", "get_restraint_set"):
            method = getattr(wrapper, getter, None)
            if callable(method):
                self._restraints.append(method())
                break
        else:
            raise TypeError(
                f"{type(wrapper).__name__} exposes neither get_restraint() nor "
                "get_restraint_set(); cannot add it to the scoring function."
            )

        self._summary[family] = self._summary.get(family, 0) + 1

    # -- restraint families -------------------------------------------------

    def connectivity(self, scale=1.0, resolution=None):
        """
        One ConnectivityRestraint per molecule copy.

        This is what stops a molecule from being torn apart into disconnected
        beads; it is applied per copy because connectivity is a property of a
        single chain, not of the assembly.
        """
        for (name, copy_index), mol in sorted(self.built.molecules.items()):
            kwargs = {"scale": scale}
            if resolution is not None:
                kwargs["resolution"] = resolution
            restraint = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(
                mol, **kwargs
            )
            restraint.set_label(f"{name}_copy{copy_index}")
            self._register(restraint, "connectivity")
        if self.verbose:
            print(f"  connectivity restraints: {self._summary.get('connectivity', 0)} "
                  f"(one per molecule copy)")
        return self

    def excluded_volume(self, resolution=1, kappa=1.0):
        """
        A single global ExcludedVolumeSphere over the whole hierarchy.

        One restraint covering everything is both cheaper and more correct than
        per-copy restraints: it automatically penalises copy-copy overlap as
        well as self-overlap.
        """
        restraint = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(
            included_objects=self.built.root_hier,
            resolution=resolution,
        )
        if kappa != 1.0:
            restraint.set_weight(float(kappa))
        restraint.set_label("all")
        self._register(restraint, "excluded_volume")
        if self.verbose:
            print(f"  excluded volume: 1 global restraint (resolution={resolution})")
        return self

    def distances(self, specs: Sequence[DistanceRestraintSpec], resolution=1):
        """
        Expand and add harmonic distance restraints.

        Each spec is expanded over copies according to its `scope`, so the
        number of restraints scales automatically with SystemSpec.n_copies.
        """
        n_copies = self.built.n_copies
        n_added = 0

        for spec in specs:
            pairs = spec.copy_pairs(n_copies)
            if not pairs:
                if self.verbose:
                    print(f"  distance spec '{spec.default_label(0, 0)}' skipped "
                          f"(scope={spec.scope} needs >= 2 copies, have {n_copies})")
                continue

            particle_pairs = []
            for copy_i, copy_j in pairs:
                p1 = select_one_particle(
                    self.built.root_hier, spec.molecule1, spec.residue1,
                    copy_index=copy_i, resolution=resolution,
                )
                p2 = select_one_particle(
                    self.built.root_hier, spec.molecule2, spec.residue2,
                    copy_index=copy_j, resolution=resolution,
                )
                if p1 == p2:
                    raise ValueError(
                        f"Distance restraint endpoints resolve to the same particle "
                        f"({spec.molecule1}{spec.residue1} and "
                        f"{spec.molecule2}{spec.residue2}, copies {copy_i}/{copy_j}). "
                        "At resolution 1 both residues may fall in the same bead."
                    )
                particle_pairs.append((p1, p2))

            label = spec.label or (
                f"{spec.molecule1}{spec.residue1}_{spec.molecule2}{spec.residue2}"
                f"_{spec.scope}"
            )
            wrapper = HarmonicDistanceRestraint(
                self.built.model,
                particle_pairs=particle_pairs,
                distance=spec.distance,
                kappa=spec.kappa,
                label=label,
            )
            self._register(wrapper, "harmonic_distance")
            n_added += len(particle_pairs)

            if self.verbose:
                print(f"  distance '{label}': {len(particle_pairs)} restraint(s), "
                      f"d0={spec.distance} k={spec.kappa}, scope={spec.scope}")

        self._summary["harmonic_distance_pairs"] = (
            self._summary.get("harmonic_distance_pairs", 0) + n_added
        )
        return self

    def custom(self, wrapper, family="custom"):
        """Escape hatch: register any PMI-style restraint object."""
        self._register(wrapper, family)
        return self

    # -- finalise -----------------------------------------------------------

    def finalize(self):
        """Build the ScoringFunction and return the bundle."""
        if not self._restraints:
            raise RuntimeError("No restraints were added before finalize().")

        scoring_function = IMP.core.RestraintsScoringFunction(self._restraints)
        bundle = RestraintBundle(
            wrappers=list(self._wrappers),
            restraints=list(self._restraints),
            scoring_function=scoring_function,
            summary=dict(self._summary),
        )
        if self.verbose:
            bundle.describe()
        return bundle
