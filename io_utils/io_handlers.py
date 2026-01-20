"""
IO utilities for saving/loading simulation data.
"""
import h5py
import numpy as np
import jax.numpy as jnp
from datetime import datetime
from typing import Dict, Any

def save_mcmc_to_hdf5(
    positions: np.ndarray,
    log_probs: np.ndarray,
    acceptance_rate: float,
    filename: str,
    system_template, # ParticleSystem
    params: Dict[str, Any] = None
):
    """
    Save MCMC trajectory to HDF5.
    """
    n_samples = positions.shape[0]
    
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
        
        # We need to unflatten each frame. 
        # Doing this loop in python might be slow for huge trajectories, 
        # but fine for toy models.
        
        # Pre-allocate arrays
        coord_buffers = {}
        for k in system_template.identity_order:
            n_copy = int(system_template.types[k]['copy'])
            coord_buffers[k] = np.zeros((n_samples, n_copy, 3))
            
        # Process frames
        for i in range(n_samples):
            # Using system_template.unflatten logic, but on numpy array
            flat = positions[i]
            idx = 0
            for k in system_template.identity_order:
                n = int(system_template.types[k]['copy'])
                coord_buffers[k][i] = flat[idx : idx + n]
                idx += n
                
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
