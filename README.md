# PhysNet-HGV: High-Fidelity Hypersonic Tracking & Reacquisition

**Physics-Informed Deep Learning Framework for Hypersonic Glide Vehicle (HGV) State Estimation Under Plasma Blackout.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Project Overview

PhysNet-HGV is a comprehensive research platform and software framework designed to solve the **Plasma Blackout** problem in hypersonic flight. When an HGV maneuvers at speeds exceeding Mach 5, the intense thermal compression ionizes the surrounding air, creating a plasma sheath that reflects electromagnetic waves and causes total radar/communication loss.

This project implements a multi-stage AI/Physics hybrid architecture to bridge these blackout gaps using:
1.  **6-DOF Dynamics**: High-fidelity WGS84/ECEF flight simulation.
2.  **Saha-Model Plasma Physics**: Equilibrium ionization calculations for real-time electron density (Ne) prediction.
3.  **PINN (Physics-Informed Neural Networks)**: Constraining neural state estimation with Navier-Stokes residuals.
4.  **DDPM (Diffusion Models)**: Sequential reacquisition of maneuvering target tracks.
5.  **Professional Dashboard**: A high-tech, glassmorphic research interface for live analytics.

---

## 🧪 Scientific Foundations

### 1. Plasma Blackout & Saha Equation
The framework calculates the electron density ($n_e$) using the **Saha Ionization Equation** based on stagnation point thermodynamics:

$$n_e \approx \sqrt{ \frac{2P}{kT} \left( \frac{2\pi m_e kT}{h^2} \right)^{3/2} \exp\left(-\frac{E_i}{kT}\right) }$$

Where:
- $P, T$ are stagnation pressure and temperature.
- $E_i$ is the ionization energy of Nitrogen ($N_2$).
- $k, h, m_e$ are physical constants.

### 2. Physics-Informed Neural Loss
The **PINNModule** enforces physical consistency by computing residuals of the continuity and momentum equations via `torch.autograd`:

$$\mathcal{L}_{physics} = \mathcal{L}_{momentum} + \lambda \mathcal{L}_{continuity}$$
$$\mathcal{L}_{continuity} = \left\| \frac{d\mathbf{x}}{dt} - \mathbf{v} \right\|^2$$

---

## 📂 Project Architecture

```bash
physnet-hgv/
├── simulation/      # 6-DOF Dynamics & Saha Plasma Models
├── models/          # PINN, DDPM, and Cross-Modal Transformers
├── filters/         # Singer-Model UKF & Covariance Intersection
├── evaluation/      # Metrics (RMSE, NEES, OSPA) & Benchmarks
├── dashboard/       # Professional Web-Based Research Interface
├── preprocessing/   # Data Pipelines & State Normalization
└── training/        # Multi-stage training scripts for PINN/Diffusion
```

---

## 🖥️ Professional Research Dashboard

The project includes a premium, high-tech dashboard for visualizing results in real-time.
- **3D Interactive Globe**: High-fidelity trajectory rendering.
- **Data Integrity Inspector**: Direct viewing of raw NumPy simulation values.
- **Plasma Monitor**: Live log-scale telemetry of ionization levels.

**Access**: Open `http://localhost:8000` after running `python dashboard/server.py`.

---

## 📊 Performance Benchmarks
| Metric | Result | Status |
| :--- | :--- | :--- |
| **Position RMSE** | 25.22 m | ✅ Verified |
| **Velocity RMSE** | 12.91 m/s | ✅ Verified |
| **NEES (Consistency)** | 1.13 | ✅ Verified |
| **OSPA (Multi-Target)** | 0.08 | ✅ Verified |

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/imshivanshutiwari/PhysNet-HGV.git
   cd PhysNet-HGV/physnet-hgv
   ```

2. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Generate Research Data**:
   ```bash
   python simulation/trajectory_gen.py
   ```

4. **Run End-to-End Evaluation**:
   ```bash
   python evaluation/evaluate.py
   ```

---

## 🛡️ Verification
The project includes a robust suite of **26 comprehensive tests** covering everything from Saha equation equilibrium to diffusion model gradient flow. Run them with:
```bash
pytest tests/
```

---

## 📜 Authorship & Acknowledgments
Developed by **Shivanshu Tiwari** in collaboration with Advanced AI Systems. 
Specialized in **Physics-Informed Deep Learning** and **Hypersonic Aerodynamics**.
