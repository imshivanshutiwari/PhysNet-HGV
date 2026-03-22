"""
Tracking Uncertainty Visualization.

Renders confidence ellipsoids and standard deviation 'envelopes' in 
3D space to represent the filter's state estimation uncertainty.
"""

import numpy as np
import plotly.graph_objects as go
from typing import List

class UncertaintyVisualizer:
    """
    Renders tracking uncertainty in 3D.
    """
    
    def plot_confidence_cone(self, trajectory: np.ndarray, covariances: np.ndarray):
        """
        trajectory: (N, 3) 
        covariances: (N, 3, 3) Positional cov
        """
        fig = go.Figure()
        
        # 1. Trajectory line
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
            mode='lines', line=dict(color='red', width=2),
            name='Estimate'
        ))
        
        # 2. Uncertainty Ellipsoids (Sparse sampling for performance)
        for i in range(0, len(trajectory), 20):
            pos = trajectory[i]
            cov = covariances[i]
            
            # Eigenvalues and Eigenvectors for ellipsoid
            vals, vecs = np.linalg.eigh(cov)
            # 3-sigma Scaling
            radii = 3.0 * np.sqrt(np.maximum(vals, 0))
            
            # Parametric Sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_s = np.outer(np.cos(u), np.sin(v))
            y_s = np.outer(np.sin(u), np.sin(v))
            z_s = np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Rotate and translate
            ellipsoid = np.stack([x_s.flatten() * radii[0], y_s.flatten() * radii[1], z_s.flatten() * radii[2]])
            ellipsoid = vecs @ ellipsoid + pos.reshape(-1, 1)
            
            fig.add_trace(go.Mesh3d(
                x=ellipsoid[0], y=ellipsoid[1], z=ellipsoid[2],
                alphahull=0, opacity=0.3, color='red', showlegend=False
            ))
            
        fig.update_layout(title="HGV Tracking with 3-Sigma Uncertainty")
        return fig

if __name__ == "__main__":
    print("UncertaintyVisualizer initialized.")
