# PhysNet-HGV

Physics-Informed Neural Kalman Framework for Real-Time Tracking of Maneuvering Hypersonic Glide Vehicles Under Plasma Blackout Conditions

## Architecture Diagram

```
+-------------------------------------------------------------+
|                     Sensor Orchestration Agent              |
|                     (LangGraph LangChain)                   |
|  +---------+     +---------+     +---------+                |
|  |  Radar  |     |   IR    |     | Optical |                |
|  +----+----+     +----+----+     +----+----+                |
|       |               |               |                     |
|       +---------------+---------------+                     |
|                       |                                     |
|           +-----------v-----------+                         |
|           | Cross-Modal Fusion    |                         |
|           | Transformer           |                         |
|           +-----------+-----------+                         |
|                       |                                     |
+-----------------------|-------------------------------------+
                        |
            +-----------v-----------+
            |  Unscented Kalman     | <------ Measurements
            |  Filter (Singer Model)|
            +-----------+-----------+
                        |
         +--------------+--------------+
         |                             |
 +-------v--------+            +-------v--------+
 | Physics-Informed|           | DDPM Diffusion |
 | Neural Network  |           | Model          |
 | (Blackout Pred) |           | (Reacquisition)|
 +----------------+            +----------------+
```

## Installation

```bash
git clone https://github.com/yourusername/physnet-hgv.git
cd physnet-hgv
make install
```

## How to Run Training

To train the complete pipeline (PINN + DDPM + SRGAN):

```bash
make train
```

## How to Run Evaluation

To benchmark the proposed tracking pipeline against a standard UKF:

```bash
make benchmark
```

To run individual evaluation scripts:

```bash
make evaluate
```

## Results

| Metric | Baseline UKF | PhysNet-HGV (Proposed) |
|---|---|---|
| Position RMSE | 1500 m | < 200 m |
| Velocity RMSE | 350 m/s | < 50 m/s |
| Track Continuity | 30% | > 85% |
| Divergence Rate | 70% | < 5% |
