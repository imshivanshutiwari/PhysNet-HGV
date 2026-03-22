"""
High-Fidelity 3D Trajectory Visualization.

Uses Plotly to generate interactive 3D visualizations of ground truth 
trajectories, estimated tracks, and maneuvering points for HGV flight.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, List

class TrajectoryPlotter:
    """
    3D visualization engine for HGV trajectories.
    """
    
    def __init__(self, earth_radius: float = 6378137.0):
        self.Re = earth_radius

    def plot_trajectories(self, truth: np.ndarray, estimates: Optional[np.ndarray] = None, title: str = "HGV Trajectory"):
        """
        truth: (N, 3) ECEF coordinates
        estimates: (N, 3) ECEF coordinates
        """
        fig = go.Figure()
        
        # 1. Ground Truth
        fig.add_trace(go.Scatter3d(
            x=truth[:, 0], y=truth[:, 1], z=truth[:, 2],
            mode='lines', line=dict(color='blue', width=4),
            name='Ground Truth'
        ))
        
        # 2. Estimates
        if estimates is not None:
            fig.add_trace(go.Scatter3d(
                x=estimates[:, 0], y=estimates[:, 1], z=estimates[:, 2],
                mode='lines', line=dict(color='red', width=3, dash='dash'),
                name='Estimated Track'
            ))
            
        # 3. Add Earth Sphere (Subtle)
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_e = self.Re * np.outer(np.cos(u), np.sin(v))
        y_e = self.Re * np.outer(np.sin(u), np.sin(v))
        z_e = self.Re * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x_e, y=y_e, z=z_e,
            opacity=0.1, colorscale='Blues', showscale=False, name='Earth'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig

if __name__ == "__main__":
    # Test Visualization
    plotter = TrajectoryPlotter()
    t = np.linspace(0, 10, 100)
    truth = np.zeros((100, 3))
    truth[:, 0] = 6378137.0 + 50000.0 + 1000 * t
    truth[:, 1] = 500 * t
    
    fig = plotter.plot_trajectories(truth)
    fig.write_html("trajectory_demo.html")
    print("Trajectory visualization saved to trajectory_demo.html")
