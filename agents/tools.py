"""
Core Tracking Tools for Autonomous Agents.

Provides the primitive operations for sensor data acquisition, 
filter updates, and physics-informed blackout bridging used 
by the LangGraph orchestration agent.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
from simulation.radar_simulator import RadarSimulator
from filters.ukf_tracker import UKFTracker
from models.pinn_module import PINNModule

class TrackingTools:
    """
    Interface for agent-driven tracking operations.
    """
    
    def __init__(self, radar: RadarSimulator, tracker: UKFTracker, pinn: PINNModule):
        self.radar = radar
        self.tracker = tracker
        self.pinn = pinn

    def get_radar_measurement(self, state: np.ndarray, t: float) -> Dict[str, Any]:
        """Tool: Acquisition of radar [R, Az, El, Doppler]."""
        z, snr, is_blackout = self.radar.get_measurement(state, t)
        return {
            "measurement": z,
            "snr_db": snr,
            "is_blackout": is_blackout
        }

    def run_ukf_prediction(self) -> Dict[str, np.ndarray]:
        """Tool: Propagate filter state forward."""
        self.tracker.predict()
        return {
            "state_pred": self.tracker.x,
            "cov_pred": self.tracker.P
        }

    def run_ukf_update(self, z: np.ndarray) -> Dict[str, np.ndarray]:
        """Tool: Incorporate new measurement."""
        # Mapping for Radar: ECEF to AER
        def h_func(x):
            r = np.linalg.norm(x[0:3] - self.radar.pos_radar)
            v_rad = np.dot(x[3:6], (x[0:3] - self.radar.pos_radar)) / r if r > 1e-3 else 0
            return np.array([r, 0, 0, v_rad])
            
        self.tracker.update(z, h_func)
        return {
            "state_post": self.tracker.x,
            "cov_post": self.tracker.P
        }

    def bridge_with_pinn(self, t: float) -> Dict[str, Any]:
        """Tool: Use Physics-Informed Neural Network to bridge blackout."""
        t_tensor = torch.tensor([[t]], dtype=torch.float32)
        with torch.no_grad():
            pinn_out = self.pinn(t_tensor).squeeze().cpu().numpy()
            
        # Update UKF with PINN pseudo-measurement
        # Assume a moderate confidence for PINN during blackout
        pinn_cov = np.eye(9) * 200.0 
        self.tracker.update_pinn_blackout(pinn_out[:9], pinn_cov)
        
        return {
            "pinn_state": pinn_out,
            "success": True
        }

if __name__ == "__main__":
    print("TrackingTools loaded.")
