# DECODE Shot Simulator

Offline trajectory simulator for **FTC DECODE 2025–26**, with full Magnus + drag + gravity physics. Built in the spirit of FRC Team **1690 Orbit's** Mo Sim — engineering-grade telemetry interface, no fluff.

```
decode_sim/
├── backend/
│   ├── physics.py       # Core RK4 trajectory engine
│   └── server.py        # Flask API
└── frontend/
    ├── index.html
    ├── styles.css
    └── app.js
```

## What it does

| Feature | Notes |
|---|---|
| **Live trajectory** | Drag-and-Magnus integrated at 1 ms with RK4. Side-view canvas with apex marker, goal catch zone, robot silhouette. |
| **Allowable error map** | Sweeps ∆v ∈ [−1, +1] m/s × ∆angle ∈ [−5°, +5°] around your nominal shot. Cyan = make. Bigger blob = more forgiving. |
| **Optimal solver** | For a target distance, finds the (velocity, angle) pair that maximizes the make-window (i.e., the most *robust* shot, not just the one that goes in once). |
| **Less-backspin study** | Drop spin via the slider and watch the tolerance map shrink/grow. Quantifies the trade-off between Magnus lift and shot consistency. |
| **Toggle drag/Magnus** | A/B comparison vs vacuum projectile to see how much each effect actually matters. |

## Physics model

- **Gravity**: g = 9.80665 m/s²
- **Drag**: F_d = ½ · ρ · C_d(Re) · A · v², piecewise C_d (≈0.47 sub-critical, drag-crisis ramp ~Re=2×10⁵). Artifact at FTC speeds sits firmly in the sub-critical regime.
- **Magnus**: F_l = ½ · ρ · C_l(S) · A · v², with C_l = S / (1 + 2S), spin parameter S = ωR/v. Saturates ~0.5, matches sport-ball wind-tunnel data.
- **Artifact spec** (DECODE 2025–26): ⌀ 5 in (0.127 m), 0.165 lb (74.8 g).
- **Goal**: AprilTag center at 29.5 in above tile floor (≈0.749 m).

## Run

```bash
cd backend
pip install flask flask-cors
python server.py
```

Open <http://localhost:5000>.

## Workflow

1. Set goal distance, launch height (your robot's release point), and starting velocity/angle.
2. Hit **SOLVE OPTIMAL SHOT** — the solver picks the pair with the widest tolerance.
3. Hit **RECOMPUTE TOLERANCE MAP** to visualize allowable error around that shot.
4. Drop **BACKSPIN** toward zero and recompute — observe whether your make-window grows (less Magnus = more predictable) or shrinks (you needed that lift). The right answer depends on distance.

## Why less backspin?

For close shots (< ~1.5 m), backspin can over-lift the ball and make it sail. Lower spin gives a flatter, more deterministic arc with smaller ∆v sensitivity. The solver + tolerance map quantify the crossover.

## Calibrating to your robot

Edit `backend/physics.py`:
- `RHO_AIR` — adjust for venue altitude (e.g., 1.07 kg/m³ at Denver).
- `cd_sphere` — if you measure your artifact has a noticeably different drag, scale here.
- `cl_magnus` coefficients — refine `a, b` empirically by fitting to your shooter's measured trajectories.
