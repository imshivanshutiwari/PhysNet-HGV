# PhysNet-HGV

**Physics-Informed Neural Kalman Framework for Real-Time Tracking of Maneuvering Hypersonic Glide Vehicles Under Plasma Blackout Conditions**

---

## 1. Overview
PhysNet-HGV is a state-of-the-art tracking framework designed to mitigate the effects of plasma blackout during hypersonic flight. It combines high-fidelity 6-DOF dynamics with Physics-Informed Neural Networks (PINN) and Generative Diffusion Models to provide continuous state estimation.

## 2. Key Features
- **Physics-Informed Neural ODE**: Captures complex plasma interactions.
- **Unscented Kalman Filter + Singer Model**: Advanced maneuvering target tracking.
- **DDPM Diffusion**: Reacquires tracks after extended blackout periods.
- **LangGraph Orchestration**: Intelligent multi-sensor switching.

## 3. Architecture
The system consists of the following components:
- `simulation`: High-fidelity WGS84 dynamics and Saha plasma models.
- `models`: Neural architectures (PINN, Transformer, SRGAN).
- `filters`: Nonlinear estimators (UKF).
- `agents`: Autonomous sensor routing.

## 4. Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## 5. Usage
To generate data:
```bash
python physnet-hgv/simulation/trajectory_gen.py
```
To train:
```bash
make train
```

## 6. Simulation
Full 6-DOF dynamics integrated with a Saha-based plasma model for realistic signal attenuation.

## 7. Deep Learning Models
Includes PINN with momentum and continuity losses, and SRGAN for optical enhancement.

## 8. Tracking Filters
Unscented Kalman Filter (UKF) utilizing the Singer acceleration model.

## 9. Diffusion & Reacquisition
DDPM implementation for bridging long blackout gaps.

## 10. Autonomous Agents
Multi-sensor orchestration using LangGraph state machines.

## 11. Evaluation
Comprehensive metrics including OSPA, GOSPA, and Track Continuity.

## 12. Deployment
TensorRT optimization for Jetson/Edge hardware.

## 13. Testing
26 pytest tests covering all critical modules.

## 14. License
MIT License.
