"""
RMF3 conversion utilities for trajectory visualization.
"""
import h5py
import numpy as np
import json
from pathlib import Path

# IMP imports - optional, only needed for RMF3 conversion
try:
    import IMP
    import IMP.core
    import IMP.atom
    import IMP.algebra
    import IMP.rmf
    import RMF
    IMP_AVAILABLE = True
except ImportError:
    IMP_AVAILABLE = False
    IMP = None


def convert_hdf5_to_rmf3(
    hdf5_file: str,
    rmf3_file: str,
    radius: float = 1.0,
    color=None,
    color_map=None,
):
    """
    Convert an HDF5 trajectory to RMF3 format for visualization in ChimeraX/PyMOL.

    Expected HDF5 layout (as written by save_mcmc_to_hdf5):
      - coordinates/{type_name}: (n_frames, n_copies, 3)
      - log_probabilities: (n_frames,)
      - system_info/: group with particle type metadata

    Args:
        hdf5_file: input HDF5 trajectory file
        rmf3_file: output RMF3 file
        radius: fallback scalar radius if not found in HDF5
        color: fallback IMP.display.Color
        color_map: dict mapping type name to RGB tuple (0-1 floats)
    
    Raises:
        ImportError: If IMP is not installed (required for RMF3 conversion)
    """
    if not IMP_AVAILABLE:
        raise ImportError(
            "IMP is not installed. RMF3 conversion requires IMP.\n"
            "On Linux: conda install -c salilab imp\n"
            "On Mac: IMP must be installed separately (see https://integrativemodeling.org/download.html)\n"
            "Note: The rest of the package works without IMP."
        )

    if color is None:
        color = IMP.display.Color(0.2, 0.6, 1.0)  # default blue

    print(f"Converting {hdf5_file} to RMF3 format...")

    with h5py.File(hdf5_file, 'r') as f:
        # Read system info
        system_info = {}
        if 'system_info' in f:
            for attr_name in f['system_info'].attrs:
                system_info[attr_name] = f['system_info'].attrs[attr_name]
        
        # Read coordinates for each particle type
        coords_grp = f['coordinates']
        particle_types = list(coords_grp.keys())
        
        # Build flat coordinate array and metadata
        all_coords = []
        particle_type_ids = []
        particle_radii = []
        particle_names = []
        
        type_id = 0
        for ptype in particle_types:
            coords_data = coords_grp[ptype][:]  # (n_frames, n_copies, 3)
            n_frames, n_copies, _ = coords_data.shape
            
            # Get radius for this type
            radius_key = f"{ptype}_radius"
            type_radius = system_info.get(radius_key, radius)
            
            for copy_idx in range(n_copies):
                all_coords.append(coords_data[:, copy_idx, :])
                particle_type_ids.append(type_id)
                particle_radii.append(type_radius)
                particle_names.append(f"{ptype}_{copy_idx}")
            
            type_id += 1
        
        # Stack into (n_frames, n_particles, 3)
        coords = np.stack(all_coords, axis=1)
        n_frames, n_particles, _ = coords.shape
        
        log_probs = f['log_probabilities'][:] if 'log_probabilities' in f else None

    # Build color lookup per particle
    def default_palette(i: int) -> IMP.display.Color:
        palette = [
            (0.2, 0.6, 1.0),  # blue
            (0.9, 0.4, 0.2),  # orange
            (0.3, 0.8, 0.4),  # green
            (0.8, 0.6, 0.2),  # yellow
            (0.6, 0.4, 0.8),  # purple
            (0.2, 0.8, 0.8),  # cyan
        ]
        r, g, b = palette[i % len(palette)]
        return IMP.display.Color(r, g, b)

    def color_for_type(tid: int, tname: str) -> IMP.display.Color:
        if color_map is not None and isinstance(color_map, dict):
            if tname in color_map:
                r, g, b = color_map[tname]
                return IMP.display.Color(float(r), float(g), float(b))
            if tid in color_map:
                r, g, b = color_map[tid]
                return IMP.display.Color(float(r), float(g), float(b))
        return default_palette(tid)

    # Map particle type names
    type_name_map = {i: ptype for i, ptype in enumerate(particle_types)}
    particle_colors = [
        color_for_type(particle_type_ids[i], type_name_map[particle_type_ids[i]]) 
        for i in range(n_particles)
    ]

    # Create IMP model/hierarchy
    model = IMP.Model()
    p_root = IMP.Particle(model)
    root_h = IMP.atom.Hierarchy.setup_particle(p_root)
    p_root.set_name("root")

    particles = []
    for i in range(n_particles):
        p = IMP.Particle(model)
        p.set_name(particle_names[i])

        xyzr = IMP.core.XYZR.setup_particle(p)
        coord0 = coords[0, i]
        xyzr.set_coordinates(IMP.algebra.Vector3D(float(coord0[0]), float(coord0[1]), float(coord0[2])))
        xyzr.set_radius(float(particle_radii[i]))
        xyzr.set_coordinates_are_optimized(True)

        IMP.atom.Mass.setup_particle(p, 1.0)
        IMP.display.Colored.setup_particle(p, particle_colors[i])

        h = IMP.atom.Hierarchy.setup_particle(p)
        root_h.add_child(h)
        particles.append(p)

    rmf = RMF.create_rmf_file(rmf3_file)
    desc = f"Trajectory: {n_frames} frames, {n_particles} particles"
    if log_probs is not None:
        desc += f", logp range [{np.min(log_probs):.2f}, {np.max(log_probs):.2f}]"
    rmf.set_description(desc)

    IMP.rmf.add_hierarchy(rmf, root_h)
    IMP.rmf.add_restraints(rmf, [])

    print(f"Writing {n_frames} frames...")
    for frame_idx in range(n_frames):
        if frame_idx % 100 == 0 or frame_idx == n_frames - 1:
            print(f"  Frame {frame_idx+1}/{n_frames}")

        for i, p in enumerate(particles):
            coord = coords[frame_idx, i]
            xyzr = IMP.core.XYZR(p)
            xyzr.set_coordinates(IMP.algebra.Vector3D(float(coord[0]), float(coord[1]), float(coord[2])))

        model.update()
        IMP.rmf.save_frame(rmf, f"frame_{frame_idx}")

    rmf.close()
    del rmf

    print(f"\n{'='*70}")
    print("RMF3 conversion complete!")
    print(f"Saved: {rmf3_file}")
    print(f"{'='*70}\n")


def inspect_hdf5(hdf5_file: str):
    """Quick inspection of HDF5 trajectory file."""
    print(f"{'='*70}")
    print(f"Inspecting: {hdf5_file}")
    print(f"{'='*70}\n")
    
    with h5py.File(hdf5_file, 'r') as f:
        print("Attributes:")
        for key, val in f.attrs.items():
            print(f"  {key}: {val}")
        
        print("\nGroups and Datasets:")
        def print_tree(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: {obj.shape} {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  {name}/")
        
        f.visititems(print_tree)
        
        if 'log_probabilities' in f:
            log_probs = f['log_probabilities'][:]
            print(f"\nLog Probabilities:")
            print(f"  min: {np.min(log_probs):.2f}")
            print(f"  max: {np.max(log_probs):.2f}")
            print(f"  mean: {np.mean(log_probs):.2f}")
        
        if 'coordinates' in f:
            coords_grp = f['coordinates']
            print(f"\nCoordinates by type:")
            for ptype in coords_grp.keys():
                shape = coords_grp[ptype].shape
                print(f"  {ptype}: {shape}")


# =============================================================================
# Sampler-agnostic trajectory writers
# -----------------------------------------------------------------------------
# Everything below works for the output of *any* BlackJAX sampler.  The only
# contract is an ``xyz`` array of shape ``(n_frames, n_particles, 3)`` -- decode
# your sampler's flat positions into that shape (e.g. with
# ``sampling.wrapper_imp_blackjax.decode_positions_to_xyz``) and hand it over.
# =============================================================================

def _broadcast_radii(radii, n_particles: float, fallback: float = 1.0) -> np.ndarray:
    """Return a ``(n_particles,)`` radius array from a scalar / array / None."""
    if radii is None:
        return np.full((n_particles,), float(fallback))
    radii = np.asarray(radii, dtype=float).ravel()
    if radii.size == 1:
        return np.full((n_particles,), float(radii[0]))
    if radii.size != n_particles:
        raise ValueError(
            f"radii has {radii.size} entries but there are {n_particles} particles"
        )
    return radii


def save_xyz_h5(
    h5_file: str,
    xyz,
    radii=None,
    names=None,
    log_probs=None,
    verbose: bool = True,
):
    """Write an ``(n_frames, n_particles, 3)`` trajectory to a simple HDF5 file.

    The layout is intentionally minimal and self-describing so it can be
    re-loaded or converted to RMF3 later (see :func:`xyz_h5_to_rmf3`):

      - ``coordinates``     : float64  ``(n_frames, n_particles, 3)``
      - ``radii``           : float64  ``(n_particles,)``
      - ``log_probabilities``: float64 ``(n_frames,)``   (optional)
      - ``names``           : str       ``(n_particles,)`` (optional)

    Works for *any* sampler -- it never imports IMP.
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be (n_frames, n_particles, 3); got {xyz.shape}")
    n_frames, n_particles, _ = xyz.shape

    out = Path(h5_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, "w") as f:
        f.attrs["n_frames"] = n_frames
        f.attrs["n_particles"] = n_particles
        f.create_dataset("coordinates", data=xyz, compression="gzip")
        f.create_dataset("radii", data=_broadcast_radii(radii, n_particles))
        if log_probs is not None:
            f.create_dataset("log_probabilities",
                             data=np.asarray(log_probs, dtype=float),
                             compression="gzip")
        if names is not None:
            dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset("names", data=np.asarray(names, dtype=object), dtype=dt)

    if verbose:
        print(f"Saved xyz trajectory ({n_frames} frames, {n_particles} particles) to {out}")


def _build_particles(model, n_particles, radii, names):
    """Create fresh XYZR particles under a root hierarchy (no source model)."""
    root_p = IMP.Particle(model)
    root_p.set_name("trajectory")
    root_h = IMP.atom.Hierarchy.setup_particle(root_p)

    radii = _broadcast_radii(radii, n_particles)
    particles = []
    for i in range(n_particles):
        p = IMP.Particle(model)
        p.set_name(names[i] if names is not None else f"bead_{i}")
        IMP.core.XYZR.setup_particle(
            p, IMP.algebra.Sphere3D(IMP.algebra.Vector3D(0.0, 0.0, 0.0), float(radii[i]))
        )
        IMP.atom.Mass.setup_particle(p, 1.0)
        root_h.add_child(IMP.atom.Hierarchy.setup_particle(p))
        particles.append(p)
    return root_h, particles


def _reuse_particles(model, particle_indexes, radii):
    """Wrap existing IMP particles (by index) in a fresh root hierarchy.

    This keeps each particle's real XYZR radius, so the RMF3 spheres match the
    sizes you set up in your IMP model.
    """
    root_p = IMP.Particle(model)
    root_p.set_name("trajectory")
    root_h = IMP.atom.Hierarchy.setup_particle(root_p)

    fallback = _broadcast_radii(radii, len(particle_indexes))
    particles = []
    for i, idx in enumerate(particle_indexes):
        p = model.get_particle(IMP.ParticleIndex(int(idx)))
        if not IMP.core.XYZR.get_is_setup(p):
            IMP.core.XYZR.setup_particle(
                p,
                IMP.algebra.Sphere3D(IMP.algebra.Vector3D(0.0, 0.0, 0.0), float(fallback[i])),
            )
        # RMF requires every leaf to carry a Mass; add a nominal one if missing.
        if not IMP.atom.Mass.get_is_setup(p):
            IMP.atom.Mass.setup_particle(p, 1.0)
        h = IMP.atom.Hierarchy(p) if IMP.atom.Hierarchy.get_is_setup(p) \
            else IMP.atom.Hierarchy.setup_particle(p)
        root_h.add_child(h)
        particles.append(p)
    return root_h, particles


def write_xyz_trajectory_rmf3(
    rmf3_file: str,
    xyz,
    imp_model=None,
    particle_indexes=None,
    radii=None,
    names=None,
    verbose: bool = True,
):
    """Write an ``(n_frames, n_particles, 3)`` trajectory straight to RMF3.

    Two modes
    ---------
    * **Reuse IMP particles** (``imp_model`` + ``particle_indexes``):
      the existing particles -- and therefore their true radii -- are written,
      one frame per slice of ``xyz``.  Use this when you already built an IMP
      model (it is the IMP-native path).
    * **Fresh particles** (otherwise): a throwaway ``IMP.Model`` is created with
      beads of the supplied ``radii``.  Use this when you only have coordinates.

    Works for the output of any sampler; ``xyz`` is all it needs.
    """
    if not IMP_AVAILABLE:
        raise ImportError(
            "IMP is not installed. RMF3 writing requires IMP/RMF.\n"
            "On Linux: conda install -c salilab imp"
        )

    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be (n_frames, n_particles, 3); got {xyz.shape}")
    n_frames, n_particles, _ = xyz.shape

    out = Path(rmf3_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    if imp_model is not None and particle_indexes is not None:
        model = imp_model
        root_h, particles = _reuse_particles(model, particle_indexes, radii)
    else:
        model = IMP.Model()
        root_h, particles = _build_particles(model, n_particles, radii, names)

    if len(particles) != n_particles:
        raise ValueError(
            f"{len(particles)} particles vs {n_particles} coordinate columns -- "
            "particle_indexes must match the decoded xyz layout"
        )

    rmf = RMF.create_rmf_file(str(out))
    rmf.set_description(f"Trajectory: {n_frames} frames, {n_particles} particles")
    IMP.rmf.add_hierarchy(rmf, root_h)
    IMP.rmf.add_restraints(rmf, [])

    for frame_idx in range(n_frames):
        frame = xyz[frame_idx]
        for i, p in enumerate(particles):
            IMP.core.XYZR(p).set_coordinates(
                IMP.algebra.Vector3D(float(frame[i, 0]), float(frame[i, 1]), float(frame[i, 2]))
            )
        model.update()
        IMP.rmf.save_frame(rmf, f"frame_{frame_idx}")
        if verbose and (frame_idx % 100 == 0 or frame_idx == n_frames - 1):
            print(f"  RMF3 frame {frame_idx + 1}/{n_frames}")

    rmf.close()
    del rmf
    if verbose:
        print(f"Saved RMF3 trajectory to {out}")


class RMF3TrajectoryWriter:
    """Stream frames to an RMF3 file *during* sampling.

    The instance is callable with the signature BlackJAX-style step callbacks
    use -- ``writer(step, position, log_prob, accepted)`` -- so it can be passed
    straight to :func:`sampling.rmh.run_rmh_sampling` (or any custom loop) to
    append one frame every ``stride`` steps without buffering the whole chain.

    Parameters
    ----------
    rmf3_file
        Output path.
    decode_xyz
        Callable ``flat_position -> (n_particles, 3)`` (e.g.
        ``adapter.decode_xyz``).
    imp_model, particle_indexes
        If both given, reuse those IMP particles (true radii).
    radii, names
        Used when building fresh particles.
    stride
        Append a frame every ``stride`` steps (default 1 = every step).

    Use as a context manager so the file is always closed::

        with RMF3TrajectoryWriter("traj.rmf3", adapter.decode_xyz,
                                  imp_model=m, particle_indexes=[i0, i1]) as w:
            run_rmh_sampling(..., step_callback=w)
    """

    def __init__(self, rmf3_file, decode_xyz, imp_model=None,
                 particle_indexes=None, radii=None, names=None, stride=1):
        if not IMP_AVAILABLE:
            raise ImportError("IMP is not installed; cannot stream RMF3 frames.")
        self.decode_xyz = decode_xyz
        self.stride = max(1, int(stride))
        self._frame = 0
        self._initialised = False

        self._rmf3_file = str(rmf3_file)
        Path(self._rmf3_file).parent.mkdir(parents=True, exist_ok=True)

        self._imp_model = imp_model
        self._particle_indexes = particle_indexes
        self._radii = radii
        self._names = names
        self._rmf = None
        self._particles = None

    def _lazy_init(self, n_particles):
        if self._imp_model is not None and self._particle_indexes is not None:
            self._model = self._imp_model
            root_h, self._particles = _reuse_particles(
                self._model, self._particle_indexes, self._radii)
        else:
            self._model = IMP.Model()
            root_h, self._particles = _build_particles(
                self._model, n_particles, self._radii, self._names)
        self._rmf = RMF.create_rmf_file(self._rmf3_file)
        IMP.rmf.add_hierarchy(self._rmf, root_h)
        IMP.rmf.add_restraints(self._rmf, [])
        self._initialised = True

    def __call__(self, step, position, log_prob=None, accepted=None):
        if step % self.stride != 0:
            return
        frame = np.asarray(self.decode_xyz(position), dtype=float)
        if not self._initialised:
            self._lazy_init(frame.shape[0])
        for i, p in enumerate(self._particles):
            IMP.core.XYZR(p).set_coordinates(
                IMP.algebra.Vector3D(float(frame[i, 0]), float(frame[i, 1]), float(frame[i, 2]))
            )
        self._model.update()
        IMP.rmf.save_frame(self._rmf, f"frame_{self._frame}")
        self._frame += 1

    def close(self):
        if self._rmf is not None:
            self._rmf.close()
            del self._rmf
            self._rmf = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def xyz_h5_to_rmf3(h5_file: str, rmf3_file: str, imp_model=None,
                   particle_indexes=None, verbose: bool = True):
    """Convert a file written by :func:`save_xyz_h5` into RMF3."""
    with h5py.File(h5_file, "r") as f:
        xyz = f["coordinates"][:]
        radii = f["radii"][:] if "radii" in f else None
        names = [n.decode() if isinstance(n, bytes) else n
                 for n in f["names"][:]] if "names" in f else None
    write_xyz_trajectory_rmf3(rmf3_file, xyz, imp_model=imp_model,
                              particle_indexes=particle_indexes,
                              radii=radii, names=names, verbose=verbose)