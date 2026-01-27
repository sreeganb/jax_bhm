#!/usr/bin/env python
"""
Plot IMP vs IMP-JAX benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO

# Set style for publication-quality figures
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42  # Ensures fonts are editable in Illustrator

# Data
data = """n_xls,n_configs,t_imp,t_impjax
30,500,7.309,4.799
60,500,9.020,6.504
90,500,9.816,7.442
120,500,9.719,7.132
200,500,9.768,7.231
500,500,9.848,7.141"""

df = pd.read_csv(StringIO(data))

# Calculate speedup
df['speedup'] = df['t_imp'] / df['t_impjax']

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Color palette
colors = {'IMP': '#2ecc71', 'IMP-JAX': '#e74c3c'}

# --- Plot 1: Timing comparison ---
ax1 = axes[0]

# Reshape data for grouped bar plot
df_melt = df.melt(
    id_vars=['n_xls'], 
    value_vars=['t_imp', 't_impjax'],
    var_name='Method', 
    value_name='Time (s)'
)
df_melt['Method'] = df_melt['Method'].map({'t_imp': 'IMP', 't_impjax': 'IMP-JAX'})

# Bar plot
bar_width = 0.35
x = np.arange(len(df))

bars1 = ax1.bar(x - bar_width/2, df['t_imp'], bar_width, 
                label='IMP (CPU)', color=colors['IMP'], edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x + bar_width/2, df['t_impjax'], bar_width, 
                label='IMP-JAX (GPU)', color=colors['IMP-JAX'], edgecolor='black', linewidth=0.5)

ax1.set_xlabel('Number of Crosslinks', fontweight='bold')
ax1.set_ylabel('Time (seconds)', fontweight='bold')
ax1.set_title('Scoring Time: 500 Configurations', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df['n_xls'])
ax1.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black')
ax1.set_ylim(0, 12)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

# --- Plot 2: Speedup ---
ax2 = axes[1]

ax2.plot(df['n_xls'], df['speedup'], 'o-', color='#3498db', linewidth=2, 
         markersize=8, markeredgecolor='black', markeredgewidth=0.5)
ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='No speedup')

ax2.fill_between(df['n_xls'], 1.0, df['speedup'], alpha=0.3, color='#3498db')

ax2.set_xlabel('Number of Crosslinks', fontweight='bold')
ax2.set_ylabel('Speedup (IMP / IMP-JAX)', fontweight='bold')
ax2.set_title('GPU Speedup Factor', fontweight='bold')
ax2.set_ylim(0.8, 1.8)

# Add value labels
for i, (x_val, y_val) in enumerate(zip(df['n_xls'], df['speedup'])):
    ax2.annotate(f'{y_val:.2f}x',
                xy=(x_val, y_val),
                xytext=(0, 8), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add mean speedup annotation
mean_speedup = df['speedup'].mean()
ax2.axhline(y=mean_speedup, color='#e74c3c', linestyle=':', linewidth=1.5)
ax2.annotate(f'Mean: {mean_speedup:.2f}x', 
            xy=(df['n_xls'].max(), mean_speedup),
            xytext=(-10, -15), textcoords="offset points",
            ha='right', va='top', fontsize=9, color='#e74c3c', fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save to PDF
output_file = 'benchmark_imp_vs_jax.pdf'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved plot to: {output_file}")

# Also save PNG for quick viewing
plt.savefig('benchmark_imp_vs_jax.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved plot to: benchmark_imp_vs_jax.png")

plt.show()