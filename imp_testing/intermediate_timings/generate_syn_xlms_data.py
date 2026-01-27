#!/usr/bin/env python
"""
Generate synthetic DSSO bifunctional crosslinks between Rpt1 and Rpt2.

DSSO crosslinker properties:
- Reacts with lysine residues
- Spacer arm length: ~10.1 Å
- Typical Cα-Cα distance cutoff: 30 Å (with flexibility)

Output: CSV file with crosslink pairs
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from Bio import PDB
    from scipy.stats import skewnorm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install biopython scipy")
    exit(1)


class DSSCrosslinkGenerator:
    """Generate synthetic DSSO crosslinks from PDB structure."""
    
    def __init__(
        self,
        pdb_file: str,
        chain_to_protein: dict,
        distance_cutoff: float = 30.0,
        skewness: float = 7.0,
        loc: float = 6.8,
        scale: float = 14.0,
    ):
        """
        Args:
            pdb_file: Path to PDB file
            chain_to_protein: Mapping from chain ID to protein name
            distance_cutoff: Maximum Cα-Cα distance for crosslinks (Å)
            skewness: Skewness parameter for distance distribution
            loc: Location parameter for skewnorm
            scale: Scale parameter for skewnorm
        """
        self.pdb_file = pdb_file
        self.chain_to_protein = chain_to_protein
        self.distance_cutoff = distance_cutoff
        self.skewness = skewness
        self.loc = loc
        self.scale = scale
        
        # Load structure
        parser = PDB.PDBParser(QUIET=True)
        self.structure = parser.get_structure('structure', pdb_file)
        
    def get_lysine_residues(self, target_chains: list = None) -> list:
        """Extract lysine residues from specified chains."""
        lysines = []
        
        for model in self.structure:
            for chain in model:
                chain_id = chain.id
                
                # Filter by target chains if specified
                if target_chains and chain_id not in target_chains:
                    continue
                
                protein_name = self.chain_to_protein.get(chain_id, f"Chain_{chain_id}")
                
                for residue in chain:
                    if residue.get_resname() == 'LYS':
                        if 'CA' in residue:
                            lysines.append({
                                'residue': residue,
                                'chain_id': chain_id,
                                'protein': protein_name,
                                'resid': residue.get_id()[1],
                                'ca_coord': residue['CA'].get_coord()
                            })
        
        print(f"Found {len(lysines)} lysine residues")
        return lysines
    
    def distance_probability(self, distance: float) -> float:
        """Probability of observing a crosslink at given distance."""
        return skewnorm.pdf(distance, self.skewness, loc=self.loc, scale=self.scale)
    
    def generate_crosslinks(
        self,
        lysines: list,
        n_crosslinks: int = 50,
        inter_protein_only: bool = True,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate synthetic crosslinks.
        
        Args:
            lysines: List of lysine residue info dicts
            n_crosslinks: Target number of crosslinks
            inter_protein_only: Only generate inter-protein crosslinks
            seed: Random seed
            
        Returns:
            DataFrame with crosslink data
        """
        np.random.seed(seed)
        
        # Compute all valid pairs with distances
        valid_pairs = []
        
        for i, lys1 in enumerate(lysines):
            for j, lys2 in enumerate(lysines):
                if j <= i:
                    continue
                
                # Skip intra-protein if requested
                if inter_protein_only and lys1['protein'] == lys2['protein']:
                    continue
                
                # Compute distance
                distance = np.linalg.norm(lys1['ca_coord'] - lys2['ca_coord'])
                
                if distance <= self.distance_cutoff:
                    prob = self.distance_probability(distance)
                    valid_pairs.append({
                        'protein1': lys1['protein'],
                        'residue1': lys1['resid'],
                        'chain1': lys1['chain_id'],
                        'protein2': lys2['protein'],
                        'residue2': lys2['resid'],
                        'chain2': lys2['chain_id'],
                        'distance': distance,
                        'probability': prob,
                    })
        
        print(f"Found {len(valid_pairs)} valid pairs within {self.distance_cutoff} Å")
        
        if not valid_pairs:
            print("No valid pairs found!")
            return pd.DataFrame()
        
        # Convert to DataFrame and sample
        df_pairs = pd.DataFrame(valid_pairs)
        
        # Normalize probabilities
        df_pairs['norm_prob'] = df_pairs['probability'] / df_pairs['probability'].sum()
        
        # Sample crosslinks based on probability
        n_to_sample = min(n_crosslinks, len(df_pairs))
        sampled_indices = np.random.choice(
            len(df_pairs),
            size=n_to_sample,
            replace=False,
            p=df_pairs['norm_prob'].values
        )
        
        df_crosslinks = df_pairs.iloc[sampled_indices].copy()
        df_crosslinks = df_crosslinks.reset_index(drop=True)
        
        print(f"Generated {len(df_crosslinks)} crosslinks")
        print(f"Distance range: {df_crosslinks['distance'].min():.1f} - {df_crosslinks['distance'].max():.1f} Å")
        print(f"Mean distance: {df_crosslinks['distance'].mean():.1f} Å")
        
        return df_crosslinks


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic DSSO crosslinks")
    parser.add_argument("--pdb", type=str, required=True, help="Input PDB file")
    parser.add_argument("--output", type=str, default="synthetic_crosslinks.csv", help="Output CSV file")
    parser.add_argument("--n-crosslinks", type=int, default=50, help="Number of crosslinks")
    parser.add_argument("--distance-cutoff", type=float, default=30.0, help="Max Cα-Cα distance (Å)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--proteins", type=str, nargs="+", default=["Rpt1", "Rpt2"], 
                        help="Proteins to include")
    args = parser.parse_args()
    
    # Chain mapping for proteasome base
    chain_to_protein = {
        'v': 'Rpt1',
        'w': 'Rpt2',
        'y': 'Rpt3',
        'z': 'Rpt4',
        '0': 'Rpt5',
        'x': 'Rpt6',
        '1': 'Rpn2',
    }
    
    # Reverse mapping to get chains for requested proteins
    protein_to_chain = {v: k for k, v in chain_to_protein.items()}
    target_chains = [protein_to_chain[p] for p in args.proteins if p in protein_to_chain]
    
    print("=" * 60)
    print("Synthetic DSSO Crosslink Generator")
    print("=" * 60)
    print(f"PDB file: {args.pdb}")
    print(f"Target proteins: {args.proteins}")
    print(f"Target chains: {target_chains}")
    print(f"Distance cutoff: {args.distance_cutoff} Å")
    print(f"Requested crosslinks: {args.n_crosslinks}")
    
    # Generate crosslinks
    generator = DSSCrosslinkGenerator(
        pdb_file=args.pdb,
        chain_to_protein=chain_to_protein,
        distance_cutoff=args.distance_cutoff,
    )
    
    lysines = generator.get_lysine_residues(target_chains=target_chains)
    
    df_crosslinks = generator.generate_crosslinks(
        lysines=lysines,
        n_crosslinks=args.n_crosslinks,
        inter_protein_only=True,
        seed=args.seed,
    )
    
    if df_crosslinks.empty:
        print("No crosslinks generated!")
        return
    
    # Save output
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full data
    df_crosslinks.to_csv(args.output, index=False)
    print(f"\nSaved crosslinks to: {args.output}")
    
    # Also save IMP-compatible format
    imp_format = df_crosslinks[['protein1', 'residue1', 'protein2', 'residue2']].copy()
    imp_format.columns = ['Protein1', 'Residue1', 'Protein2', 'Residue2']
    imp_output = str(args.output).replace('.csv', '_imp_format.csv')
    imp_format.to_csv(imp_output, index=False)
    print(f"Saved IMP format to: {imp_output}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Crosslink Summary")
    print("=" * 60)
    print(df_crosslinks[['protein1', 'residue1', 'protein2', 'residue2', 'distance']].to_string())


if __name__ == "__main__":
    main()