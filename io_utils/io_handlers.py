"""
IO utilities for saving/loading simulation data.
"""
import h5py
import numpy as np
import jax.numpy as jnp
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

def save_mcmc_to_hdf5(
    positions: np.ndarray,
    log_probs: np.ndarray,
    acceptance_rate: float,
    filename: str,
    system_template, # ParticleSystem
    params: Dict[str, Any] = None,
    convert_to_rmf3: bool = False,
    color_map: Dict[str, tuple] = None
):
    """
    Save MCMC trajectory to HDF5, optionally convert to RMF3.
    
    Args:
        positions: Array of particle positions (n_samples, n_particles*3)
        log_probs: Log probability for each sample
        acceptance_rate: Overall acceptance rate
        filename: Output HDF5 filename
        system_template: ParticleSystem instance
        params: Additional parameters to save
        convert_to_rmf3: If True, also create RMF3 file (requires IMP)
        color_map: Optional color mapping for RMF3 visualization
    """
    n_samples = positions.shape[0]
    
    # Ensure output directory exists
    output_path = Path(filename).parent
    output_path.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filename, 'w') as f:
        # Metadata
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['n_samples'] = n_samples
        f.attrs['acceptance_rate'] = acceptance_rate
        
        # Save System Info
        grp_sys = f.create_group('system_info')
        for k, v in system_template.types.items():
            grp_sys.attrs[f"{k}_radius"] = v['radius']
            grp_sys.attrs[f"{k}_copy"] = v['copy']
            
        # Coordinates Group
        coords_grp = f.create_group('coordinates')
        
        # Pre-allocate arrays
        coord_buffers = {}
        for k in system_template.identity_order:
            n_copy = int(system_template.types[k]['copy'])
            coord_buffers[k] = np.zeros((n_samples, n_copy, 3))
            
        # Process frames
        for i in range(n_samples):
            flat = positions[i]  # Shape: (n_particles * 3,)
            idx = 0
            for k in system_template.identity_order:
                n = int(system_template.types[k]['copy'])
                n_coords = n * 3  # Total coordinates for this type
                coord_buffers[k][i] = flat[idx : idx + n_coords].reshape(n, 3)
                idx += n_coords
                
        # Write to HDF5
        for k, data in coord_buffers.items():
            coords_grp.create_dataset(k, data=data, compression='gzip')
            
        # Log Probs
        f.create_dataset('log_probabilities', data=log_probs, compression='gzip')
        
        # Best Config
        best_idx = np.argmax(log_probs)
        best_grp = f.create_group('best_configuration')
        best_grp.attrs['sample_index'] = best_idx
        best_grp.attrs['log_probability'] = float(log_probs[best_idx])
        
        for k in coord_buffers:
            best_grp.create_dataset(k, data=coord_buffers[k][best_idx])
        
        # Params
        if params:
            p_grp = f.create_group('parameters')
            for k, v in params.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        p_grp.attrs[f"{k}_{sub_k}"] = sub_v
                else:
                    try:
                        p_grp.attrs[k] = v
                    except:
                        pass # Skip non-serializable

    print(f"Saved trajectory to {filename}")
    
    # Convert to RMF3 if requested
    if convert_to_rmf3:
        try:
            from .rmf3_converter import convert_hdf5_to_rmf3
            rmf3_filename = str(Path(filename).with_suffix('.rmf3'))
            convert_hdf5_to_rmf3(filename, rmf3_filename, color_map=color_map)
        except ImportError as e:
            print(f"Warning: Could not convert to RMF3: {e}")
            print("Install IMP to enable RMF3 conversion: conda install -c salilab imp")