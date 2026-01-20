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