# DECODE Shot Simulator

Offline trajectory simulator for **FTC DECODE 2025–26**. Gravity + quadratic aerodynamic drag,
integrated with RK4 and exact Hermite event location.

```
Trajectory-Simulation/
├── backend/
│   ├── physics.py        # RK4 engine, cached surrogate, make-window solver
│   ├── server.py         # Flask API
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── styles.css
    └── app.js
```

## What it does

| Feature | Notes |
|---|---|
| **Live trajectory** | RK4 with cubic-Hermite event location. Side-view canvas with apex marker, goal catch zone, robot silhouette. |
| **Tolerance sweep** | Draws representative perturbations around the nominal command plus make/miss boundary trajectories. |
| **Tolerance map** | Plots the scoring region in (∆angle, ∆velocity) space with nested bounded perturbation boxes. |
| **Robust coverage** | Deterministic made fraction over the declared bounded perturbation box. It is a robustness score, not a fitted probability. |
| **Optimal solver** | Finds the command whose nearby bounded perturbations have the highest simulated make fraction. The cached event map accelerates evaluation without changing that rule. |
| **Shot family** | Every commandable shot that scores at the current distance — one per launch angle, each at the speed with the widest make window — drawn as a single fan coloured by robustness. Hover for the numbers, click to load one into the launch controls. |
| **Sampled cloud** | N uniform samples from the bounded box, drawn made/missed beside deterministic grid coverage. |
| **Target sweep** | The most robust shot at every distance in a range, plotted as exit velocity and launch angle against distance, with a CSV export for a distance-indexed shooter table. |
| **Drag on/off** | A/B comparison against a vacuum projectile to see how much drag actually matters. |
| **CSV lookup table** | Exports velocity × angle → made/miss + impact data for use on the robot. |

## Physics model

- **Gravity**: g = 9.80665 m/s²
- **Drag**: F_d = ½ · ρ · C_d · A · v² with the effective constant C_d = 0.47 used by the implementation.
- **Magnus / spin: NOT modelled.** `ShotParams` still carries `spin_rpm` and `enable_magnus` for API
  compatibility, but the equations of motion ignore both. This is gravity + drag only.
- **Artifact spec** (DECODE 2025–26): ⌀ 5 in (0.127 m), 0.165 lb (74.8 g).
- **Goal**: scoring opening top lip at **38.75 in (0.98425 m)**, depth 18.3 in (0.46482 m). The
  29.5 in figure sometimes quoted is the AprilTag centre height, *not* the scoring opening.
- **Scoring**: the ball scores when its centre crosses the top-lip plane while descending, with x
  inside the opening after shrinking the front and back lips by one artifact radius.

## Run

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python server.py
```

Open <http://localhost:5000>.

## Workflow

1. Set goal distance, launch height (your robot's release point), and velocity/angle.
2. Set the **VELOCITY ±** and **ANGLE ±** half-widths from measured release-state variation,
   quantization, latency, and the engineering tolerance the command should survive.
3. Hit **SOLVE OPTIMAL SHOT** — returns the most robust shot for that distance, along with how much
   slack you have in each direction.
4. Hit **RUN SWEEP**, then open the **TOLERANCE MAP** tab to see the scoring region against the
   bounded perturbation box. If the region is lopsided, bias your shooter toward the wide side.
5. Hit **RUN SHOT FAMILY** to see every shot that scores at this distance at once. The fan is
   coloured by robustness against the best margin actually available, so the greenest curve is
   the one to shoot even when the whole family is marginal. Click a curve or a row in **ALL
   SHOTS** to load it into the launch controls.
6. Switch the fan to **CLOUD** to sample the bounded box. Sampled make fraction is shown beside
   deterministic grid coverage as a visualization check, not a calibrated success probability.
7. **RUN TARGET SWEEP** solves the most robust shot at every distance in the range and exports
   it as CSV — this is the table to index by vision distance on the robot.
8. **DOWNLOAD LOOKUP TABLE** to bake the results into your robot code.

## Family vs cloud

They look alike and answer different questions. The **family** shows a robust scoring command at
each launch angle. The **cloud** is a sensitivity visualization: uniform samples inside the
declared engineering box around one command. It shows what the selected bounds imply without
assuming a Gaussian noise model.

## How the solver works

`x_at_top` is the horizontal position where the ball descends through the rim plane. The simulator
integrates a velocity-angle lattice once, caches this target-independent event map, and thresholds it
for each requested goal distance. A two-dimensional prefix sum then counts every rectangular
perturbation window efficiently. The selected command remains the make-window command: a nominally
scoring candidate with the highest simulated made fraction in its nearby box.

Event location uses a cubic Hermite root solve rather than linear interpolation. Because velocity is
known at both ends of a step, the crossing stays accurate to ~1e-6 m even at dt = 0.02 s, which is
what lets the integrator run coarse. The root is found by binary search over the step: the step
already brackets the crossing, so bisection converges unconditionally, with no derivative to divide
by and no way to run away on a nearly flat segment.

## Calibrating to your robot

Edit `backend/physics.py`:
- `RHO_AIR` — adjust for venue altitude (e.g. 1.07 kg/m³ at Denver).
- `_cd` / `_cd_np` — if your artifact measures a noticeably different drag, scale here. **Edit both**;
  `_cd` is the readable reference and `_cd_np` is what runs.
- `DT_SURROGATE` / `DT_PATH` — integration step for the solver and for drawn arcs respectively.

After changing any physical constant, restart the server — the surrogate cache is keyed on launch
geometry, not on the constants.
