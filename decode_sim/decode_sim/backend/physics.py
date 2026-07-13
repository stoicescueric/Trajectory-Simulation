"""
DECODE Artifact Trajectory Physics Engine
==========================================
Simulates a 5" FTC DECODE artifact ball with gravity + quadratic drag.
Magnus effect excluded.

Coordinate system (2D side view):
  x : horizontal (m), positive toward goal
  y : vertical   (m), positive up

GOAL model — ball enters through the TOP opening:
  GOAL_TOP_HEIGHT_M = 0.98425 m (38.75 in) is the actual top lip per the
  DECODE 2025-26 game manual.  The 29.5 in figure is the AprilTag centre Z,
  NOT the scoring opening.

  A ball SCORES when its center crosses y = goal_height while descending
  (vy < 0), with x inside the top opening after shrinking the front/back
  lips by one artifact radius.
"""

from dataclasses import dataclass, field
import math
from typing import List, Optional
import numpy as np


# ── Physical constants ──────────────────────────────────────────────────────
G          = 9.80665               # m/s²
RHO_AIR    = 1.204                 # kg/m³  (20 °C, sea level)
ARTIFACT_D = 0.127                 # m  (5 in)
ARTIFACT_R = ARTIFACT_D / 2.0
ARTIFACT_M = 0.0748                # kg (0.165 lb)
ARTIFACT_A = math.pi * ARTIFACT_R ** 2
NU_AIR     = 1.516e-5              # m²/s

# ── Official DECODE 2025-26 goal geometry ───────────────────────────────────
GOAL_TOP_HEIGHT_M = 0.98425        # 38.75 in → top lip of basket opening
GOAL_DEPTH_M      = 0.46482        # 18.3 in  → horizontal depth of opening


def in_to_m(inches: float) -> float:  return inches * 0.0254
def m_to_in(m: float)       -> float:  return m / 0.0254


def _entry_x_bounds(goal_distance: float, goal_depth: float):
    """Center-of-ball x limits for clearing the front/back lips."""
    return goal_distance + ARTIFACT_R, goal_distance + goal_depth - ARTIFACT_R


# ── Drag coefficient (smooth sphere) ────────────────────────────────────────
def _cd(v: float) -> float:
    re = v * ARTIFACT_D / NU_AIR
    if re < 1.0:    return 24.0 / max(re, 1e-6)
    if re < 1e3:    return 24.0 / re + 6.0 / (1.0 + math.sqrt(re)) + 0.4
    if re < 2e5:    return 0.47          # sub-critical — where FTC speeds sit
    if re < 3.5e5:  return 0.47 - (re - 2e5) / 1.5e5 * 0.27
    return 0.20


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class ShotParams:
    velocity:      float
    angle_deg:     float
    launch_height: float            # m, above floor
    launch_x:      float = 0.0
    goal_distance: float = 1.5      # m, to FRONT FACE of basket
    goal_height:   float = GOAL_TOP_HEIGHT_M   # m, top lip
    goal_depth:    float = GOAL_DEPTH_M        # m, horizontal depth
    wind:          float = 0.0
    enable_drag:   bool  = True
    # Kept for API compatibility; physics ignores both.
    spin_rpm:      float = 0.0
    enable_magnus: bool  = False


@dataclass
class TrajectoryResult:
    t:  List[float] = field(default_factory=list)
    x:  List[float] = field(default_factory=list)
    y:  List[float] = field(default_factory=list)
    vx: List[float] = field(default_factory=list)
    vy: List[float] = field(default_factory=list)
    apex_x:           float          = 0.0
    apex_y:           float          = 0.0
    range_x:          float          = 0.0
    # y when ball crosses the front face (x = goal_distance) — display only
    impact_y_at_goal: Optional[float] = None
    # x where ball descends through the top plane — scoring event
    x_at_top:         Optional[float] = None
    made:             bool            = False
    entry_angle_deg:  Optional[float] = None


# ── Equations of motion ──────────────────────────────────────────────────────
def _accel(state, p: ShotParams):
    _, y, vx, vy = state
    if y < 0:
        return (0.0, 0.0)
    ux    = vx - p.wind
    uy    = vy
    speed = math.hypot(ux, uy)
    ax, ay = 0.0, -G
    if p.enable_drag and speed > 1e-6:
        fd  = 0.5 * RHO_AIR * _cd(speed) * ARTIFACT_A * speed * speed
        ax -= fd * ux / speed / ARTIFACT_M
        ay -= fd * uy / speed / ARTIFACT_M
    return (ax, ay)


# ── RK4 integrator ──────────────────────────────────────────────────────────
def simulate(p: ShotParams, dt: float = 0.001, t_max: float = 4.0) -> TrajectoryResult:
    angle = math.radians(p.angle_deg)
    state = [
        p.launch_x,
        p.launch_height,
        p.velocity * math.cos(angle),
        p.velocity * math.sin(angle),
    ]

    res          = TrajectoryResult()
    t            = 0.0
    apex_y       = state[1];  apex_x       = state[0]
    last_x       = state[0];  last_y       = state[1]
    crossed_front = False   # fired once when x passes goal_distance
    crossed_top   = False   # fired once when y descends through goal_height
    entry_x_min, entry_x_max = _entry_x_bounds(p.goal_distance, p.goal_depth)

    res.t.append(t); res.x.append(state[0]); res.y.append(state[1])
    res.vx.append(state[2]); res.vy.append(state[3])

    while t < t_max:
        def deriv(s):
            ax, ay = _accel(s, p)
            return [s[2], s[3], ax, ay]

        k1 = deriv(state)
        s2 = [state[i] + 0.5*dt*k1[i] for i in range(4)]
        k2 = deriv(s2)
        s3 = [state[i] + 0.5*dt*k2[i] for i in range(4)]
        k3 = deriv(s3)
        s4 = [state[i] + dt*k3[i] for i in range(4)]
        k4 = deriv(s4)
        state = [state[i] + dt/6.0*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
                 for i in range(4)]
        t += dt

        if state[1] > apex_y:
            apex_y = state[1]; apex_x = state[0]

        # ── Display: height at goal front face (x = goal_distance) ──────────
        if not crossed_front and last_x < p.goal_distance <= state[0]:
            frac = (p.goal_distance - last_x) / (state[0] - last_x + 1e-12)
            res.impact_y_at_goal = last_y + frac * (state[1] - last_y)
            crossed_front = True

        # ── Scoring: ball descends through the top plane ─────────────────────
        # Condition: was above goal_height last step, at/below it now.
        if not crossed_top and last_y > p.goal_height >= state[1]:
            frac   = (last_y - p.goal_height) / (last_y - state[1] + 1e-12)
            x_cross = last_x  + frac * (state[0] - last_x)
            vy_at   = res.vy[-1] + frac * (state[3] - res.vy[-1])
            vx_at   = res.vx[-1] + frac * (state[2] - res.vx[-1])
            res.x_at_top        = x_cross
            res.entry_angle_deg = math.degrees(math.atan2(-vy_at, abs(vx_at)))
            # Score: descending, cleared front lip, hasn't overshot back wall
            res.made = (
                vy_at < 0
                and entry_x_min <= x_cross <= entry_x_max
            )
            crossed_top = True

        res.t.append(t); res.x.append(state[0]); res.y.append(state[1])
        res.vx.append(state[2]); res.vy.append(state[3])
        last_x = state[0]; last_y = state[1]

        if state[1] <= 0 and t > 0.05:
            break
        # Stop well past the basket back wall so top-plane crossing isn't missed
        if state[0] > p.goal_distance + p.goal_depth + 2.0:
            break

    res.apex_x = apex_x; res.apex_y = apex_y; res.range_x = state[0]
    return res


# ── Tolerance sweep (spaghetti trajectories) ────────────────────────────────
def make_sweep(p: ShotParams,
               dv_range: float = 0.7,
               da_range: float = 1.5,
               n: int = 11) -> dict:
    dvs = [dv_range * (2*i/(n-1) - 1) for i in range(n)]
    das = [da_range * (2*j/(n-1) - 1) for j in range(n)]

    trajs = []
    for dv in dvs:
        for da in das:
            tp = ShotParams(
                velocity      = max(0.1, p.velocity + dv),
                angle_deg     = p.angle_deg + da,
                launch_height = p.launch_height,
                launch_x      = p.launch_x,
                goal_distance = p.goal_distance,
                goal_height   = p.goal_height,
                goal_depth    = p.goal_depth,
                wind          = p.wind,
                enable_drag   = p.enable_drag,
            )
            r    = simulate(tp, dt=0.002, t_max=3.0)
            step = max(1, len(r.t) // 80)
            trajs.append({
                "x":    r.x[::step],
                "y":    r.y[::step],
                "made": r.made,
                "impact_y_at_goal": r.impact_y_at_goal,
                "dv": dv, "da": da,
            })

    made = sum(1 for tr in trajs if tr["made"])
    nr   = simulate(p, dt=0.002, t_max=3.0)
    step = max(1, len(nr.t) // 80)

    return {
        "trajectories": trajs,
        "nominal": {"x": nr.x[::step], "y": nr.y[::step], "made": nr.made},
        "made_pct": 100.0 * made / len(trajs) if trajs else 0.0,
        "goal": {
            "distance": p.goal_distance,
            "height":   p.goal_height,
            "depth":    p.goal_depth,
        },
    }


# ── Vectorised batch simulator (numpy) ──────────────────────────────────────
def _simulate_made_batch(velocities, angles_deg,
                         launch_height: float,
                         goal_distance: float,
                         goal_height:   float = GOAL_TOP_HEIGHT_M,
                         goal_depth:    float = GOAL_DEPTH_M,
                         wind:          float = 0.0,
                         enable_drag:   bool  = True,
                         dt:            float = 0.002,
                         t_max:         float = 3.0) -> np.ndarray:
    """
    RK4 batch simulation: runs N (velocity, angle) pairs simultaneously with
    numpy arrays instead of N sequential Python loops.
    Returns a boolean ndarray made[i] for each shot.
    10-50x faster than N calls to simulate() for large N.
    """
    N    = len(velocities)
    vels = np.asarray(velocities, dtype=float)
    angs = np.radians(np.asarray(angles_deg, dtype=float))

    x  = np.zeros(N)
    y  = np.full(N, float(launch_height))
    vx = vels * np.cos(angs)
    vy = vels * np.sin(angs)
    lx = x.copy();  ly = y.copy()

    made = np.zeros(N, dtype=bool)
    done = np.zeros(N, dtype=bool)
    t    = 0.0
    entry_x_min, entry_x_max = _entry_x_bounds(goal_distance, goal_depth)

    def _accel_np(cy, cvx, cvy):
        ux = cvx - wind
        uy = cvy
        speed = np.hypot(ux, uy)
        ax_   = np.zeros(N)
        ay_   = np.where(cy >= 0, -G, 0.0)
        if enable_drag:
            re  = np.maximum(speed * ARTIFACT_D / NU_AIR, 1e-9)
            cd  = np.where(re < 1.0,   24.0 / re,
                  np.where(re < 1e3,   24.0 / re + 6.0 / (1.0 + np.sqrt(re)) + 0.4,
                  np.where(re < 2e5,   0.47,
                  np.where(re < 3.5e5, 0.47 - (re - 2e5) / 1.5e5 * 0.27, 0.20))))
            ok  = (cy >= 0) & (speed > 1e-6)
            spd = np.where(ok, speed, 1.0)
            fd  = np.where(ok, 0.5 * RHO_AIR * cd * ARTIFACT_A * speed * speed, 0.0)
            ax_ = np.where(ok, -fd * ux / spd / ARTIFACT_M, 0.0)
            ay_ = np.where(ok, -G - fd * uy / spd / ARTIFACT_M, ay_)
        return ax_, ay_

    while t < t_max:
        if done.all():
            break

        # True RK4 update, matching simulate() stage structure.
        ax1, ay1 = _accel_np(y, vx, vy)
        kx1, ky1 = vx, vy

        vx2 = vx + 0.5 * dt * ax1
        vy2 = vy + 0.5 * dt * ay1
        y2  = y  + 0.5 * dt * ky1
        ax2, ay2 = _accel_np(y2, vx2, vy2)
        kx2, ky2 = vx2, vy2

        vx3 = vx + 0.5 * dt * ax2
        vy3 = vy + 0.5 * dt * ay2
        y3  = y  + 0.5 * dt * ky2
        ax3, ay3 = _accel_np(y3, vx3, vy3)
        kx3, ky3 = vx3, vy3

        vx4 = vx + dt * ax3
        vy4 = vy + dt * ay3
        y4  = y  + dt * ky3
        ax4, ay4 = _accel_np(y4, vx4, vy4)
        kx4, ky4 = vx4, vy4

        nx  = x + (dt / 6.0) * (kx1 + 2*kx2 + 2*kx3 + kx4)
        ny  = y + (dt / 6.0) * (ky1 + 2*ky2 + 2*ky3 + ky4)
        nvx = vx + (dt / 6.0) * (ax1 + 2*ax2 + 2*ax3 + ax4)
        nvy = vy + (dt / 6.0) * (ay1 + 2*ay2 + 2*ay3 + ay4)

        # Descending top-plane crossing
        cross = (~done) & (ly > goal_height) & (ny <= goal_height)
        if cross.any():
            frac    = (ly - goal_height) / np.maximum(ly - ny, 1e-12)
            x_cross = lx + frac * (nx - lx)
            vy_at   = vy + frac * (nvy - vy)
            in_goal = ((entry_x_min <= x_cross) &
                       (x_cross <= entry_x_max) &
                       (vy_at < 0))
            made   |= cross & in_goal
            done   |= cross

        done |= (~done) & (
            ((ny <= 0) & (t > 0.05)) |
            (nx > goal_distance + goal_depth + 2.0)
        )

        lx[:] = x;  ly[:] = y
        x, y, vx, vy = nx, ny, nvx, nvy
        t += dt

    return made


# ── Optimal solver ───────────────────────────────────────────────────────────
def _find_optimal_window_scan(goal_distance: float,
                              launch_height: float,
                              enable_drag: bool = True,
                              dv_range: float = 0.7,
                              da_range: float = 1.5) -> Optional[ShotParams]:
    """
    Find the (velocity, angle) pair that maximises the make-window fraction
    within the tolerance box (±dv_range m/s, ±da_range degrees).

    Both phases use _simulate_made_batch (numpy vectorised RK4), running all
    trajectories as one array operation rather than N sequential Python loops.

    Phase 1 — velocity window scan  (56 batch calls, N=131 each)
        For each angle (1° steps, 25–80°) run all velocities in one batch.
        Find scoring interval [v_lo, v_hi]; midpoint is v_center.
        Rank candidates by window width (v_hi − v_lo).

    Phase 2 — sweep evaluation  (up to 20 batch calls, N=81 each)
        For the top-20 candidates, evaluate the 9x9 perturbation grid in a
        single batch call. Return the shot with the highest make-fraction.
    """
    V_STEP      = 0.10
    V_LO, V_HI = 2.0, 15.0
    N_V         = round((V_HI - V_LO) / V_STEP) + 1
    vels        = [round(V_LO + V_STEP * i, 2) for i in range(N_V)]

    # ── Phase 1: batch scan per angle ────────────────────────────────
    candidates = []   # (v_window, v_center, angle)

    for a_int in range(25, 81):    # 56 angles, 1° steps
        a    = float(a_int)
        made = _simulate_made_batch(
            vels, [a] * N_V,
            launch_height, goal_distance,
            goal_height=GOAL_TOP_HEIGHT_M,
            goal_depth=GOAL_DEPTH_M,
            enable_drag=enable_drag,
        )
        made_vs = [v for v, m in zip(vels, made) if m]
        if len(made_vs) < 2:
            continue
        v_window = made_vs[-1] - made_vs[0]
        v_center = (made_vs[0] + made_vs[-1]) / 2.0
        candidates.append((v_window, v_center, a))

    if not candidates:
        return None

    candidates.sort(reverse=True)          # widest velocity window first

    # ── Phase 2: batch sweep for top-20 candidates ───────────────────
    N_SWEEP = 9
    dvs = [dv_range * (2 * i / (N_SWEEP - 1) - 1) for i in range(N_SWEEP)]
    das = [da_range * (2 * j / (N_SWEEP - 1) - 1) for j in range(N_SWEEP)]

    best     : Optional[ShotParams] = None
    best_pct : float                = -1.0

    for _, v_center, a in candidates[:20]:
        s_v = [max(0.1, v_center + dv) for dv in dvs for _ in das]
        s_a = [a + da                  for _  in dvs for da in das]

        made = _simulate_made_batch(
            s_v, s_a,
            launch_height, goal_distance,
            goal_height=GOAL_TOP_HEIGHT_M,
            goal_depth=GOAL_DEPTH_M,
            enable_drag=enable_drag,
        )
        pct = float(made.sum()) / len(made) * 100.0
        if pct > best_pct:
            best_pct = pct
            best = ShotParams(velocity=v_center, angle_deg=a,
                              launch_height=launch_height,
                              goal_distance=goal_distance,
                              enable_drag=enable_drag)

    return best


def _step_grid(lo: float, hi: float, step: float) -> np.ndarray:
    count = int(math.floor((hi - lo) / step + 1e-9)) + 1
    return np.round(lo + step * np.arange(count), 10)


def _best_make_window_on_grid(v_grid: np.ndarray,
                              a_grid: np.ndarray,
                              made_grid: np.ndarray,
                              dv_range: float,
                              da_range: float,
                              v_min: float,
                              v_max: float,
                              a_min: float,
                              a_max: float):
    made_grid = np.asarray(made_grid, dtype=bool)
    prefix = np.pad(made_grid.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    valid_v = np.flatnonzero(
        (v_grid >= v_min) &
        (v_grid <= v_max) &
        (v_grid - dv_range >= v_grid[0] - 1e-9) &
        (v_grid + dv_range <= v_grid[-1] + 1e-9)
    )
    valid_a = np.flatnonzero(
        (a_grid >= a_min) &
        (a_grid <= a_max) &
        (a_grid - da_range >= a_grid[0] - 1e-9) &
        (a_grid + da_range <= a_grid[-1] + 1e-9)
    )

    best = None
    for ia in valid_a:
        a0 = np.searchsorted(a_grid, a_grid[ia] - da_range, side="left")
        a1 = np.searchsorted(a_grid, a_grid[ia] + da_range, side="right")
        for iv in valid_v:
            if not made_grid[iv, ia]:
                continue
            v0 = np.searchsorted(v_grid, v_grid[iv] - dv_range, side="left")
            v1 = np.searchsorted(v_grid, v_grid[iv] + dv_range, side="right")
            total = (v1 - v0) * (a1 - a0)
            hits = prefix[v1, a1] - prefix[v0, a1] - prefix[v1, a0] + prefix[v0, a0]
            pct = 100.0 * float(hits) / float(total)

            key = (pct, int(hits), -float(v_grid[iv]))
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "pct": pct,
                    "hits": int(hits),
                    "total": int(total),
                    "velocity": float(v_grid[iv]),
                    "angle_deg": float(a_grid[ia]),
                }
    return best


def find_optimal(goal_distance: float,
                 launch_height: float,
                 enable_drag: bool = True,
                 dv_range: float = 0.7,
                 da_range: float = 1.5,
                 goal_height: float = GOAL_TOP_HEIGHT_M,
                 goal_depth: float = GOAL_DEPTH_M,
                 wind: float = 0.0) -> Optional[ShotParams]:
    """
    Maximise the 2D make-window fraction for a commanded shot.

    Commanded shots stay within the UI range (2-12 m/s, 20-80 deg). The
    perturbation window can extend outside that range because real error can
    do that too. Candidate shots only compete when their whole tolerance box
    is present in the made/miss grid, so edge candidates are not rewarded for
    having a clipped sweep.
    """
    v_min, v_max = 2.0, 12.0
    a_min, a_max = 20.0, 80.0

    v_grid = _step_grid(max(0.1, v_min - dv_range), v_max + dv_range, 0.20)
    a_grid = _step_grid(a_min - da_range, a_max + da_range, 0.50)
    vv, aa = np.meshgrid(v_grid, a_grid, indexing="ij")
    made = _simulate_made_batch(
        vv.ravel(), aa.ravel(),
        launch_height, goal_distance,
        goal_height=goal_height, goal_depth=goal_depth, wind=wind,
        enable_drag=enable_drag,
    ).reshape(len(v_grid), len(a_grid))
    coarse = _best_make_window_on_grid(
        v_grid, a_grid, made,
        dv_range, da_range,
        v_min, v_max, a_min, a_max,
    )
    if coarse is None:
        return None

    v_search_lo = max(v_min, coarse["velocity"] - 1.0)
    v_search_hi = min(v_max, coarse["velocity"] + 1.0)
    a_search_lo = max(a_min, coarse["angle_deg"] - 2.5)
    a_search_hi = min(a_max, coarse["angle_deg"] + 2.5)

    v_fine = _step_grid(max(0.1, v_search_lo - dv_range), v_search_hi + dv_range, 0.025)
    a_fine = _step_grid(a_search_lo - da_range, a_search_hi + da_range, 0.10)
    vv_f, aa_f = np.meshgrid(v_fine, a_fine, indexing="ij")
    made_f = _simulate_made_batch(
        vv_f.ravel(), aa_f.ravel(),
        launch_height, goal_distance,
        goal_height=goal_height, goal_depth=goal_depth, wind=wind,
        enable_drag=enable_drag,
    ).reshape(len(v_fine), len(a_fine))
    fine = _best_make_window_on_grid(
        v_fine, a_fine, made_f,
        dv_range, da_range,
        v_search_lo, v_search_hi, a_search_lo, a_search_hi,
    ) or coarse

    return ShotParams(
        velocity=fine["velocity"],
        angle_deg=fine["angle_deg"],
        launch_height=launch_height,
        goal_distance=goal_distance,
        goal_height=goal_height,
        goal_depth=goal_depth,
        wind=wind,
        enable_drag=enable_drag,
    )


# ── Lookup table ─────────────────────────────────────────────────────────────
def build_lut(p: ShotParams,
              v_min: float = 2.0,  v_max: float = 12.0,  v_steps: int = 50,
              a_min: float = 20.0, a_max: float = 80.0,  a_steps: int = 61,
              ) -> List[dict]:
    rows = []
    for iv in range(v_steps):
        v = v_min + (v_max - v_min) * iv / (v_steps - 1)
        for ia in range(a_steps):
            a = a_min + (a_max - a_min) * ia / (a_steps - 1)
            tp = ShotParams(velocity=v, angle_deg=a,
                           launch_height=p.launch_height,
                           goal_distance=p.goal_distance,
                           wind=p.wind, enable_drag=p.enable_drag)
            r = simulate(tp, dt=0.002, t_max=3.0)
            rows.append({
                "velocity_ms":      round(v, 3),
                "angle_deg":        round(a, 2),
                "made":             int(r.made),
                "x_at_top_m":       round(r.x_at_top,         4) if r.x_at_top         is not None else "",
                "impact_y_front_m": round(r.impact_y_at_goal,  4) if r.impact_y_at_goal is not None else "",
                "entry_angle_deg":  round(r.entry_angle_deg,   2) if r.entry_angle_deg  is not None else "",
                "flight_time_s":    round(r.t[-1],             4) if r.t                else "",
                "apex_y_m":         round(r.apex_y,            4),
            })
    return rows


if __name__ == "__main__":
    p = ShotParams(velocity=6.0, angle_deg=55,
                   launch_height=in_to_m(15.75),
                   goal_distance=in_to_m(60))
    r = simulate(p)
    print(f"apex:          ({r.apex_x:.3f}, {r.apex_y:.3f}) m")
    print(f"x_at_top:      {r.x_at_top}")
    print(f"impact_y_front:{r.impact_y_at_goal}")
    print(f"entry_angle:   {r.entry_angle_deg}")
    print(f"made:          {r.made}")
    best = find_optimal(in_to_m(60), in_to_m(15.75))
    if best:
        rb = simulate(best)
        print(f"\noptimal: v={best.velocity:.2f} m/s  angle={best.angle_deg:.1f}°  made={rb.made}")
