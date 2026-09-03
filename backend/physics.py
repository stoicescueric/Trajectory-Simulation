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

Design notes
------------
Everything integrates through ONE vectorised RK4 core (`_integrate`).  Two
properties of this problem make the analysis cheap:

  1. The rim crossing is located with a CUBIC HERMITE root solve — bisected
     over the step — rather than linear interpolation.  Because vy is known
     at both ends of a step, the crossing is 4th-order accurate, so the step
     size can be ~10x larger than a linearly-interpolated event would allow
     (measured: 9e-5 m crossing error at dt = 0.02, 0.04 and even 0.08).

  2. `x_at_top` — the x where the ball descends through the rim plane — does
     not depend on goal distance (distance enters only the final made/miss
     comparison). Exact velocity-angle event maps are therefore reusable
     across targets. A prefix sum evaluates every bounded make-window while
     preserving the season decision rule.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
import math
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Physical constants ──────────────────────────────────────────────────────
G          = 9.80665               # m/s²
RHO_AIR    = 1.204                 # kg/m³  (20 °C, sea level)
ARTIFACT_D = 0.127                 # m  (5 in)
ARTIFACT_R = ARTIFACT_D / 2.0
ARTIFACT_M = 0.0748                # kg (0.165 lb)
ARTIFACT_A = math.pi * ARTIFACT_R ** 2
NU_AIR     = 1.516e-5              # m²/s (used only to report the operating Re range)
CD_EFFECTIVE = 0.47                # constant effective coefficient in the measured regime

# ── Official DECODE 2025-26 goal geometry ───────────────────────────────────
GOAL_TOP_HEIGHT_M = 0.98425        # 38.75 in → top lip of basket opening
GOAL_DEPTH_M      = 0.46482        # 18.3 in  → horizontal depth of opening

# ── Integration step sizes ──────────────────────────────────────────────────
# Hermite event location makes accuracy nearly independent of dt, so the
# surrogate runs coarse.  Drawn paths use a finer step purely for a smooth
# polyline.
DT_SURROGATE = 0.02
DT_PATH      = 0.005

# ── Deployed command envelope ───────────────────────────────────────────────
# Manual plots may explore a wider range, but recommendations exported for the
# robot must stay inside the measured flywheel and hood calibration envelope.
V_MIN_CMD, V_MAX_CMD = 4.70, 6.61
A_MIN_CMD, A_MAX_CMD = 42.0, 58.0

def in_to_m(inches: float) -> float:  return inches * 0.0254
def m_to_in(m: float)       -> float:  return m / 0.0254


def _entry_x_bounds(goal_distance: float, goal_depth: float):
    """Center-of-ball x limits for clearing the front/back lips."""
    return goal_distance + ARTIFACT_R, goal_distance + goal_depth - ARTIFACT_R


def _validate_shot(p: "ShotParams") -> None:
    values = (
        p.velocity, p.angle_deg, p.launch_height, p.launch_x,
        p.goal_distance, p.goal_height, p.goal_depth, p.wind,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("trajectory inputs must be finite numbers")
    if p.velocity <= 0.0 or not (-89.9 < p.angle_deg < 89.9):
        raise ValueError("velocity must be positive and angle must lie between -89.9 and 89.9 degrees")
    if p.launch_height < 0.0 or p.goal_height < 0.0 or p.goal_distance <= 0.0:
        raise ValueError("invalid launch or goal geometry")
    if p.goal_depth <= 2.0 * ARTIFACT_R:
        raise ValueError("goal depth must exceed the artifact diameter")


# ── Effective drag coefficient ──────────────────────────────────────────────
# The measured launcher range remains on the sub-critical branch, so the
# deployed model uses one explicit effective value rather than advertising a
# Reynolds-dependent model that never changes coefficient in operation.
def _cd(v: float) -> float:
    del v
    return CD_EFFECTIVE


def _cd_np(speed: np.ndarray) -> np.ndarray:
    """Vectorised twin of `_cd`."""
    return np.full_like(np.asarray(speed, dtype=float), CD_EFFECTIVE)


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
    # time of that same rim-plane event (not the later integration stop time)
    t_at_top:         Optional[float] = None
    made:             bool            = False
    entry_angle_deg:  Optional[float] = None


# ── Equations of motion ──────────────────────────────────────────────────────
def _accel_np(y, vx, vy, wind: float, enable_drag: bool):
    ux    = vx - wind
    uy    = vy
    speed = np.hypot(ux, uy)
    above = y >= 0
    ax    = np.zeros_like(vx)
    ay    = np.where(above, -G, 0.0)
    if enable_drag:
        ok  = above & (speed > 1e-6)
        spd = np.where(ok, speed, 1.0)
        fd  = np.where(ok, 0.5 * RHO_AIR * _cd_np(speed) * ARTIFACT_A * speed * speed, 0.0)
        ax  = np.where(ok, -fd * ux / spd / ARTIFACT_M, 0.0)
        ay  = np.where(ok, -G - fd * uy / spd / ARTIFACT_M, ay)
    return ax, ay


# ── Cubic Hermite event location ────────────────────────────────────────────
# A step gives us position AND velocity at both ends, which is exactly the
# data a cubic Hermite needs.  Solving the Hermite for the crossing keeps the
# event 4th-order accurate no matter how coarse dt is.  The crossing itself is
# bracketed by the step, so a binary search finds it without any derivative.
def _hermite_eval(p0, d0, p1, d1, dt: float, tau):
    t2 = tau * tau
    t3 = t2 * tau
    return ((2*t3 - 3*t2 + 1) * p0 + (t3 - 2*t2 + tau) * dt * d0
          + (-2*t3 + 3*t2)    * p1 + (t3 - t2)         * dt * d1)


def _hermite_root(p0, d0, p1, d1, dt: float, target, iters: int = 16):
    """τ ∈ [0,1] where the Hermite through (p0,d0)→(p1,d1) equals `target`.

    Binary search.  Callers only ask on steps that already bracket the
    crossing, so halving the interval converges unconditionally — there is no
    derivative to divide by, and nothing to run away when the segment is
    nearly flat. Sixteen fixed halvings put the event-location error far below
    the RK4 discretisation error without implying spurious precision.
    """
    start = np.asarray(p0, dtype=float) - target
    lo    = np.zeros_like(start)
    hi    = np.ones_like(start)
    s0    = np.sign(start)              # sign of the residual at τ = 0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f   = _hermite_eval(p0, d0, p1, d1, dt, mid) - target
        # Same sign as the τ=0 end ⇒ the crossing is in the upper half.
        upper = np.sign(f) == s0
        lo = np.where(upper, mid, lo)
        hi = np.where(upper, hi, mid)
    return 0.5 * (lo + hi)


def _hermite_derivative(p0, d0, p1, d1, dt: float, tau):
    """Time derivative of the cubic Hermite interpolant at ``tau``."""
    t2 = tau * tau
    dp_dtau = ((6*t2 - 6*tau) * p0
               + (3*t2 - 4*tau + 1) * dt * d0
               + (-6*t2 + 6*tau) * p1
               + (3*t2 - 2*tau) * dt * d1)
    return dp_dtau / dt


# ── Vectorised RK4 core ─────────────────────────────────────────────────────
def _integrate(velocities,
               angles_deg,
               launch_height: float,
               goal_height:   float = GOAL_TOP_HEIGHT_M,
               wind:          float = 0.0,
               enable_drag:   bool  = True,
               dt:            float = DT_SURROGATE,
               t_max:         float = 4.0,
               launch_x:      float = 0.0,
               front_x:       Optional[float] = None,
               x_max:         Optional[float] = None,
               stop_at_rim:   bool  = True,
               record_path:   bool  = False) -> dict:
    """
    Integrate N shots simultaneously.

    `stop_at_rim` ends a trajectory as soon as it is below the rim and
    descending — it can never score after that, since vy only decreases.  This
    is both tighter than falling all the way to the floor AND what makes the
    result independent of goal_distance.  Pass False (with `x_max`) when the
    full arc is wanted for drawing.

    Returns arrays indexed like the inputs.  `x_at_top` is NaN for shots that
    never descend through the rim plane.
    """
    v   = np.asarray(velocities, dtype=float).ravel()
    ang = np.radians(np.asarray(angles_deg, dtype=float).ravel())
    if v.size != ang.size:
        raise ValueError("velocity and angle arrays must have equal length")
    scalars = (launch_height, goal_height, wind, dt, t_max, launch_x)
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError("integration inputs must be finite numbers")
    if dt <= 0.0 or t_max <= 0.0 or launch_height < 0.0 or goal_height < 0.0:
        raise ValueError("invalid integration interval or geometry")
    if not (np.isfinite(v).all() and np.isfinite(ang).all()) or np.any(v <= 0.0):
        raise ValueError("launch arrays must contain finite positive speeds and finite angles")
    N   = v.size

    out = {
        "x_at_top":        np.full(N, np.nan),
        "entry_angle_deg": np.full(N, np.nan),
        "t_at_top":        np.full(N, np.nan),
        "descending":      np.zeros(N, dtype=bool),
        "y_at_front":      np.full(N, np.nan),
        "apex_x":          np.full(N, float(launch_x)),
        "apex_y":          np.full(N, float(launch_height)),
        "t_end":           np.zeros(N),
        "range_x":         np.full(N, float(launch_x)),
    }
    if N == 0:
        out["paths"] = {}
        return out

    x  = np.full(N, float(launch_x))
    y  = np.full(N, float(launch_height))
    vx = v * np.cos(ang)
    vy = v * np.sin(ang)

    idx      = np.arange(N)          # active slot → original index
    finished = np.zeros(N, dtype=bool)
    paths: Dict[int, list] = {i: [] for i in range(N)} if record_path else {}
    t = 0.0

    while idx.size and t < t_max:
        if record_path:
            for slot, orig in enumerate(idx):
                paths[orig].append((t, x[slot], y[slot], vx[slot], vy[slot]))

        # ── RK4 stage structure ─────────────────────────────────────────────
        ax1, ay1 = _accel_np(y, vx, vy, wind, enable_drag)
        kx1, ky1 = vx, vy

        vx2, vy2 = vx + 0.5*dt*ax1, vy + 0.5*dt*ay1
        y2       = y  + 0.5*dt*ky1
        ax2, ay2 = _accel_np(y2, vx2, vy2, wind, enable_drag)
        kx2, ky2 = vx2, vy2

        vx3, vy3 = vx + 0.5*dt*ax2, vy + 0.5*dt*ay2
        y3       = y  + 0.5*dt*ky2
        ax3, ay3 = _accel_np(y3, vx3, vy3, wind, enable_drag)
        kx3, ky3 = vx3, vy3

        vx4, vy4 = vx + dt*ax3, vy + dt*ay3
        y4       = y  + dt*ky3
        ax4, ay4 = _accel_np(y4, vx4, vy4, wind, enable_drag)
        kx4, ky4 = vx4, vy4

        nx  = x  + (dt/6.0) * (kx1 + 2*kx2 + 2*kx3 + kx4)
        ny  = y  + (dt/6.0) * (ky1 + 2*ky2 + 2*ky3 + ky4)
        nvx = vx + (dt/6.0) * (ax1 + 2*ax2 + 2*ax3 + ax4)
        nvy = vy + (dt/6.0) * (ay1 + 2*ay2 + 2*ay3 + ay4)

        # ── Apex ────────────────────────────────────────────────────────────
        higher = ny > out["apex_y"][idx]
        if higher.any():
            hit = idx[higher]
            out["apex_y"][hit] = ny[higher]
            out["apex_x"][hit] = nx[higher]

        # ── Display: height at the goal front face ──────────────────────────
        # Both events search only the steps that bracket a crossing: the
        # bisection costs a fixed number of Hermite evaluations, so it is
        # worth shrinking what it runs on.
        if front_x is not None:
            cf = (x < front_x) & (nx >= front_x) & np.isnan(out["y_at_front"][idx])
            if cf.any():
                tau = _hermite_root(x[cf], vx[cf], nx[cf], nvx[cf], dt, front_x)
                out["y_at_front"][idx[cf]] = _hermite_eval(
                    y[cf], vy[cf], ny[cf], nvy[cf], dt, tau)

        # ── Scoring event: descending through the rim plane ─────────────────
        cross = (y > goal_height) & (ny <= goal_height) & np.isnan(out["x_at_top"][idx])
        if cross.any():
            tau = _hermite_root(y[cross], vy[cross], ny[cross], nvy[cross],
                                dt, goal_height)
            xc  = _hermite_eval(x[cross], vx[cross], nx[cross], nvx[cross], dt, tau)
            vyc = _hermite_derivative(y[cross], vy[cross], ny[cross], nvy[cross], dt, tau)
            vxc = _hermite_derivative(x[cross], vx[cross], nx[cross], nvx[cross], dt, tau)
            hit = idx[cross]
            out["x_at_top"][hit]        = xc
            out["entry_angle_deg"][hit] = np.degrees(np.arctan2(-vyc, np.abs(vxc)))
            out["descending"][hit]      = vyc < 0
            out["t_at_top"][hit]        = t + tau * dt

        x, y, vx, vy = nx, ny, nvx, nvy
        t += dt

        # ── Termination ─────────────────────────────────────────────────────
        done = (y <= 0) & (t > 0.05)
        if stop_at_rim:
            done |= (y < goal_height) & (vy < 0)
        if x_max is not None:
            done |= x > x_max

        if done.any():
            hit = idx[done]
            out["t_end"][hit]   = t
            out["range_x"][hit] = x[done]
            finished[hit]       = True
            if record_path:
                for slot in np.flatnonzero(done):
                    paths[idx[slot]].append(
                        (t, x[slot], y[slot], vx[slot], vy[slot]))
            keep = ~done
            x, y, vx, vy = x[keep], y[keep], vx[keep], vy[keep]
            idx = idx[keep]

    # Anything still flying when t_max hit.
    if idx.size:
        out["t_end"][idx]   = t
        out["range_x"][idx] = x
        if record_path:
            for slot, orig in enumerate(idx):
                paths[orig].append((t, x[slot], y[slot], vx[slot], vy[slot]))

    if record_path:
        out["paths"] = paths
    return out


def _made_from(x_at_top: np.ndarray,
               descending: np.ndarray,
               entry_x_min: float,
               entry_x_max: float) -> np.ndarray:
    """Scoring test — the only place goal_distance enters."""
    with np.errstate(invalid="ignore"):
        return (descending
                & (x_at_top >= entry_x_min)
                & (x_at_top <= entry_x_max))


# ── Path densification ──────────────────────────────────────────────────────
# A step carries position AND velocity at both ends, so the same cubic Hermite
# that locates events also reconstructs the arc between samples.  That means
# the integrator can run coarse — the drawn curve is smooth because it is
# interpolated exactly, not because it was integrated finely.
def _densify(path: list, n_out: int = 220):
    if len(path) < 2:
        cols = list(zip(*path)) if path else ([], [], [], [], [])
        return [list(c) for c in cols]

    t  = np.array([s[0] for s in path])
    x  = np.array([s[1] for s in path])
    y  = np.array([s[2] for s in path])
    vx = np.array([s[3] for s in path])
    vy = np.array([s[4] for s in path])

    tt  = np.linspace(t[0], t[-1], n_out)
    seg = np.clip(np.searchsorted(t, tt, side="right") - 1, 0, t.size - 2)
    h   = t[seg + 1] - t[seg]
    h   = np.where(h > 1e-15, h, 1e-15)
    tau = np.clip((tt - t[seg]) / h, 0.0, 1.0)

    xx = _hermite_eval(x[seg], vx[seg], x[seg + 1], vx[seg + 1], h, tau)
    yy = _hermite_eval(y[seg], vy[seg], y[seg + 1], vy[seg + 1], h, tau)
    return [tt.tolist(), xx.tolist(), yy.tolist(),
            np.interp(tt, t, vx).tolist(), np.interp(tt, t, vy).tolist()]


# ── Single-shot wrapper (keeps the original public shape) ───────────────────
def simulate(p: ShotParams, dt: float = DT_SURROGATE, t_max: float = 4.0) -> TrajectoryResult:
    _validate_shot(p)
    entry_x_min, entry_x_max = _entry_x_bounds(p.goal_distance, p.goal_depth)
    r = _integrate(
        [p.velocity], [p.angle_deg], p.launch_height,
        goal_height=p.goal_height, wind=p.wind, enable_drag=p.enable_drag,
        dt=dt, t_max=t_max, launch_x=p.launch_x,
        front_x=p.goal_distance,
        x_max=p.goal_distance + p.goal_depth + 2.0,
        stop_at_rim=False, record_path=True,
    )
    res = TrajectoryResult()
    res.t, res.x, res.y, res.vx, res.vy = _densify(r["paths"][0])

    res.apex_x  = float(r["apex_x"][0])
    res.apex_y  = float(r["apex_y"][0])
    res.range_x = float(r["range_x"][0])

    yf = r["y_at_front"][0]
    res.impact_y_at_goal = None if math.isnan(yf) else float(yf)
    xt = r["x_at_top"][0]
    res.x_at_top = None if math.isnan(xt) else float(xt)
    tt = r["t_at_top"][0]
    res.t_at_top = None if math.isnan(tt) else float(tt)
    ea = r["entry_angle_deg"][0]
    res.entry_angle_deg = None if math.isnan(ea) else float(ea)
    res.made = bool(_made_from(r["x_at_top"], r["descending"],
                               entry_x_min, entry_x_max)[0])
    return res


# ── Surrogate: x_at_top over (velocity, angle), cached ──────────────────────
# goal_distance is deliberately absent — one surrogate serves every distance.
_SURROGATE_V = (1.0, 14.0, 0.25)     # lo, hi, step — wider than the UI range
_SURROGATE_A = (12.0, 88.0, 1.0)     #                so bounded sweeps stay in-grid


@dataclass
class Surrogate:
    v_grid:    np.ndarray
    a_grid:    np.ndarray
    x_at_top:  np.ndarray            # [nv, na], NaN where it never descends through
    descending: np.ndarray
    entry_angle_deg: np.ndarray
    t_at_top:  np.ndarray
    apex_y:    np.ndarray
    monotone:  bool                  # guard-rail result


_SUR_CACHE: "OrderedDict[tuple, Surrogate]" = OrderedDict()
_SUR_LOCK  = threading.Lock()
_SUR_MAX   = 16

# Exact event-map tiles used by the current make-window selector.
# A tile stores X(v, angle), not a target-specific made/miss mask, so the same
# integration is reusable at every target distance.
_EVENT_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_EVENT_LOCK = threading.Lock()
_EVENT_MAX = 24


def _uniform(lo: float, hi: float, step: float) -> np.ndarray:
    n = int(math.floor((hi - lo) / step + 1e-9)) + 1
    return np.round(lo + step * np.arange(n), 10)


def _surrogate_key(launch_height, goal_height, wind, enable_drag, dt):
    return (round(float(launch_height), 6), round(float(goal_height), 6),
            round(float(wind), 6), bool(enable_drag), round(float(dt), 6),
            _SURROGATE_V, _SURROGATE_A)


def build_surrogate(launch_height: float,
                    goal_height:   float = GOAL_TOP_HEIGHT_M,
                    wind:          float = 0.0,
                    enable_drag:   bool  = True,
                    dt:            float = DT_SURROGATE) -> Surrogate:
    v_grid = _uniform(*_SURROGATE_V)
    a_grid = _uniform(*_SURROGATE_A)
    vv, aa = np.meshgrid(v_grid, a_grid, indexing="ij")
    r = _integrate(vv.ravel(), aa.ravel(), launch_height,
                   goal_height=goal_height, wind=wind, enable_drag=enable_drag,
                   dt=dt, stop_at_rim=True)
    shape = (v_grid.size, a_grid.size)

    xt = r["x_at_top"].reshape(shape)
    # Diagnostic: x_at_top rises with velocity at fixed angle across the whole
    # commanded range.  `band()` no longer *depends* on this (it scans for sign
    # changes), but a violation here means the physics has changed shape and is
    # worth surfacing.  A strong headwind can break it just outside the
    # commanded range, at angles too steep to shoot anyway.
    monotone = True
    in_cmd = (a_grid >= A_MIN_CMD) & (a_grid <= A_MAX_CMD)
    for j in np.flatnonzero(in_cmd):
        col = xt[:, j]
        ok  = ~np.isnan(col)
        if ok.sum() >= 2 and np.any(np.diff(col[ok]) < -1e-9):
            monotone = False
            break

    return Surrogate(
        v_grid=v_grid, a_grid=a_grid,
        x_at_top=xt,
        descending=r["descending"].reshape(shape),
        entry_angle_deg=r["entry_angle_deg"].reshape(shape),
        t_at_top=r["t_at_top"].reshape(shape),
        apex_y=r["apex_y"].reshape(shape),
        monotone=monotone,
    )


def get_surrogate(launch_height: float,
                  goal_height:   float = GOAL_TOP_HEIGHT_M,
                  wind:          float = 0.0,
                  enable_drag:   bool  = True,
                  dt:            float = DT_SURROGATE) -> Surrogate:
    key = _surrogate_key(launch_height, goal_height, wind, enable_drag, dt)
    with _SUR_LOCK:
        hit = _SUR_CACHE.get(key)
        if hit is not None:
            _SUR_CACHE.move_to_end(key)
            return hit
    # Built outside the lock; a rare duplicate build is cheaper than blocking.
    sur = build_surrogate(launch_height, goal_height, wind, enable_drag, dt)
    with _SUR_LOCK:
        _SUR_CACHE[key] = sur
        _SUR_CACHE.move_to_end(key)
        while len(_SUR_CACHE) > _SUR_MAX:
            _SUR_CACHE.popitem(last=False)
    return sur


def clear_surrogate_cache() -> None:
    with _SUR_LOCK:
        _SUR_CACHE.clear()
    with _EVENT_LOCK:
        _EVENT_CACHE.clear()


def _grid_key(values: np.ndarray) -> tuple:
    values = np.asarray(values, dtype=float).ravel()
    return tuple(np.round(values, 10).tolist())


def _event_grid(v_grid: np.ndarray,
                a_grid: np.ndarray,
                launch_height: float,
                goal_height: float,
                wind: float,
                enable_drag: bool,
                dt: float = DT_SURROGATE) -> Tuple[np.ndarray, np.ndarray]:
    """Return cached descending-crossing events on an exact command lattice."""
    v_grid = np.asarray(v_grid, dtype=float).ravel()
    a_grid = np.asarray(a_grid, dtype=float).ravel()
    key = (round(float(launch_height), 6), round(float(goal_height), 6),
           round(float(wind), 6), bool(enable_drag), round(float(dt), 6),
           _grid_key(v_grid), _grid_key(a_grid))
    with _EVENT_LOCK:
        hit = _EVENT_CACHE.get(key)
        if hit is not None:
            _EVENT_CACHE.move_to_end(key)
            return hit

    vv, aa = np.meshgrid(v_grid, a_grid, indexing="ij")
    result = _integrate(
        vv.ravel(), aa.ravel(), launch_height,
        goal_height=goal_height, wind=wind, enable_drag=enable_drag,
        dt=dt, stop_at_rim=True,
    )
    shape = (v_grid.size, a_grid.size)
    tile = (result["x_at_top"].reshape(shape),
            result["descending"].reshape(shape))
    with _EVENT_LOCK:
        _EVENT_CACHE[key] = tile
        _EVENT_CACHE.move_to_end(key)
        while len(_EVENT_CACHE) > _EVENT_MAX:
            _EVENT_CACHE.popitem(last=False)
    return tile


def _made_grid(v_grid: np.ndarray,
               a_grid: np.ndarray,
               launch_height: float,
               goal_height: float,
               goal_depth: float,
               goal_distance: float,
               wind: float,
               enable_drag: bool) -> np.ndarray:
    x_at_top, descending = _event_grid(
        v_grid, a_grid, launch_height, goal_height, wind, enable_drag)
    entry_x_min, entry_x_max = _entry_x_bounds(goal_distance, goal_depth)
    return _made_from(x_at_top, descending, entry_x_min, entry_x_max)


def _expanded_grid(lo: float, hi: float, pad: float, step: float) -> np.ndarray:
    """Uniform grid anchored at ``lo`` with enough cells to contain the box."""
    n_left = int(math.ceil(pad / step - 1e-12))
    n_core = int(math.floor((hi - lo) / step + 1e-9))
    n_right = n_left
    return np.round(lo + step * np.arange(-n_left, n_core + n_right + 1), 10)


def _best_make_window_on_grid(v_grid: np.ndarray,
                              a_grid: np.ndarray,
                              made_grid: np.ndarray,
                              dv_range: float,
                              da_range: float,
                              v_min: float,
                              v_max: float,
                              a_min: float,
                              a_max: float):
    """Season make-window count with deterministic speed/angle tie-breaking."""
    made_grid = np.asarray(made_grid, dtype=bool)
    prefix = np.pad(made_grid.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    valid_v = np.flatnonzero(
        (v_grid >= v_min - 1e-9) & (v_grid <= v_max + 1e-9)
        & (v_grid - dv_range >= v_grid[0] - 1e-9)
        & (v_grid + dv_range <= v_grid[-1] + 1e-9))
    valid_a = np.flatnonzero(
        (a_grid >= a_min - 1e-9) & (a_grid <= a_max + 1e-9)
        & (a_grid - da_range >= a_grid[0] - 1e-9)
        & (a_grid + da_range <= a_grid[-1] + 1e-9))

    best = None
    for ia in valid_a:
        a0 = int(np.searchsorted(a_grid, a_grid[ia] - da_range, side="left"))
        a1 = int(np.searchsorted(a_grid, a_grid[ia] + da_range, side="right"))
        for iv in valid_v:
            if not made_grid[iv, ia]:
                continue
            v0 = int(np.searchsorted(v_grid, v_grid[iv] - dv_range, side="left"))
            v1 = int(np.searchsorted(v_grid, v_grid[iv] + dv_range, side="right"))
            total = (v1 - v0) * (a1 - a0)
            hits = int(prefix[v1, a1] - prefix[v0, a1]
                       - prefix[v1, a0] + prefix[v0, a0])
            fraction = hits / total
            key = (fraction, hits, -float(v_grid[iv]), -float(a_grid[ia]))
            if best is None or key > best["key"]:
                best = {
                    "key": key, "fraction": fraction,
                    "hits": hits, "total": total,
                    "velocity": float(v_grid[iv]),
                    "angle_deg": float(a_grid[ia]),
                }
    return best


# ── Band extraction: invert the monotone x_at_top per angle ─────────────────
def _invert_column(v_grid: np.ndarray, col: np.ndarray, target: float,
                   prefer: str = "first") -> float:
    """
    Velocity where x_at_top == target, by linear interpolation. NaN if unreachable.

    Scans for sign changes rather than assuming a globally monotone column, so
    a local wiggle (which a strong headwind can produce at extreme angles)
    degrades to picking the outermost crossing instead of returning nonsense.
    `prefer="first"` takes the lowest such velocity, `"last"` the highest —
    together they bracket the widest scoring interval.
    """
    ok = np.flatnonzero(~np.isnan(col))
    if ok.size < 2:
        return math.nan
    vs = v_grid[ok]
    xs = col[ok]

    lo, hi = xs[:-1], xs[1:]
    hits = np.flatnonzero(((lo <= target) & (hi >= target)) |
                          ((lo >= target) & (hi <= target)))
    if hits.size == 0:
        # At the reachability knee the first finite rim crossing may already
        # lie beyond the front-lip threshold.  In that case the physical lower
        # edge is the reachability boundary; using the first finite sample is
        # conservative and avoids deleting a real feasible interval.
        if prefer == "first" and xs[0] >= target:
            return float(vs[0])
        # If the whole sampled column remains before the requested back edge,
        # the feasible interval continues to the top of the surrogate.
        if prefer == "last" and xs[-1] <= target:
            return float(vs[-1])
        return math.nan
    k = int(hits[0] if prefer == "first" else hits[-1])

    x0, x1 = xs[k], xs[k + 1]
    if abs(x1 - x0) <= 1e-15:
        return float(vs[k])
    f = (target - x0) / (x1 - x0)
    return float(vs[k] + f * (vs[k + 1] - vs[k]))


def band(sur: Surrogate,
         entry_x_min: float,
         entry_x_max: float,
         a_fine: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scoring band boundaries.  Returns (a_fine, v_lo, v_hi); NaN where the
    angle cannot score at all.
    """
    v_lo_c = np.array([_invert_column(sur.v_grid, sur.x_at_top[:, j], entry_x_min, "first")
                       for j in range(sur.a_grid.size)])
    v_hi_c = np.array([_invert_column(sur.v_grid, sur.x_at_top[:, j], entry_x_max, "last")
                       for j in range(sur.a_grid.size)])

    if a_fine is None:
        # Spans the whole surrogate, not just the commanded 20-80°: shooter
        # The bounded perturbation domain extends beyond the command envelope;
        # truncating it there would inflate the reported robust coverage.
        a_fine = _uniform(float(sur.a_grid[0]), float(sur.a_grid[-1]), 0.1)

    # Interpolate across angle, but never across a gap of infeasible angles.
    def _across(vals):
        ok = ~np.isnan(vals)
        if ok.sum() < 2:
            return np.full(a_fine.size, math.nan)
        out = np.interp(a_fine, sur.a_grid[ok], vals[ok], left=math.nan, right=math.nan)
        # Blank out fine angles that fall inside an infeasible stretch.
        near = np.interp(a_fine, sur.a_grid, ok.astype(float))
        out[near < 0.999] = math.nan
        return out

    return a_fine, _across(v_lo_c), _across(v_hi_c)


# ── Reported robustness: clearance in perturbation-half-width units ──
# Read out next to the chosen shot; it is not used by the selector.
def _margin_map(a_c: np.ndarray, v_c: np.ndarray,
                a_b: np.ndarray, v_lo: np.ndarray, v_hi: np.ndarray,
                sigma_v: float, sigma_a: float,
                window_sigma: float = 5.0) -> np.ndarray:
    """
    For each candidate (v_c[i], a_c[i]) return the distance, normalized by the
    declared perturbation half-widths, to the nearest point of the band
    boundary. Negative outside the band.

    Distance is measured to the boundary CURVES directly rather than to a
    rasterised mask, so the answer is continuous rather than pixel-limited.
    """
    n = a_c.size
    best = np.full(n, np.inf)

    # Interpolate the band at each candidate's own angle for the inside test.
    lo_here = np.interp(a_c, a_b, v_lo, left=math.nan, right=math.nan)
    hi_here = np.interp(a_c, a_b, v_hi, left=math.nan, right=math.nan)
    with np.errstate(invalid="ignore"):
        inside = (v_c >= lo_here) & (v_c <= hi_here)

    step   = float(a_b[1] - a_b[0]) if a_b.size > 1 else 1.0
    if not (math.isfinite(sigma_v) and math.isfinite(sigma_a)
            and sigma_v > 0.0 and sigma_a > 0.0):
        raise ValueError("robustness half-widths must be positive finite numbers")
    reach  = min(a_b.size, int(math.ceil(window_sigma * sigma_a / max(step, 1e-9))))
    j_home = np.clip(np.searchsorted(a_b, a_c), 0, a_b.size - 1)

    valid = ~np.isnan(v_lo) & ~np.isnan(v_hi)
    for off in range(-reach, reach + 1):
        j = np.clip(j_home + off, 0, a_b.size - 1)
        ok = valid[j]
        da = (a_c - a_b[j]) / sigma_a
        for edge in (v_lo, v_hi):
            dv = (v_c - edge[j]) / sigma_v
            d  = np.sqrt(dv * dv + da * da)
            best = np.where(ok & (d < best), d, best)

    # The feasible angle range itself is a boundary.
    if valid.any():
        a_first, a_last = a_b[valid][0], a_b[valid][-1]
        best = np.minimum(best, np.abs(a_c - a_first) / sigma_a)
        best = np.minimum(best, np.abs(a_c - a_last) / sigma_a)

    best = np.minimum(best, window_sigma)
    return np.where(inside, best, -best)


# ── Neighbourhood make-window: the original decision rule ──────────────────
def directional_margins(v: float, a: float,
                        a_b: np.ndarray, v_lo: np.ndarray, v_hi: np.ndarray) -> dict:
    """Slack in physical units, separately in each direction."""
    lo = float(np.interp(a, a_b, v_lo, left=math.nan, right=math.nan))
    hi = float(np.interp(a, a_b, v_hi, left=math.nan, right=math.nan))
    out = {
        "v_down_ms": None if math.isnan(lo) else max(0.0, v - lo),
        "v_up_ms":   None if math.isnan(hi) else max(0.0, hi - v),
        "a_down_deg": 0.0,
        "a_up_deg":   0.0,
    }

    def _scan(direction: int) -> float:
        step = 0.05 * direction
        travelled = 0.0
        while abs(travelled) < 30.0:
            travelled += step
            l = float(np.interp(a + travelled, a_b, v_lo, left=math.nan, right=math.nan))
            h = float(np.interp(a + travelled, a_b, v_hi, left=math.nan, right=math.nan))
            if math.isnan(l) or math.isnan(h) or not (l <= v <= h):
                return abs(travelled) - abs(step)
        return 30.0

    out["a_down_deg"] = _scan(-1)
    out["a_up_deg"]   = _scan(+1)
    return out


# ── Optimal solver ───────────────────────────────────────────────────────────
def find_optimal(goal_distance: float,
                 launch_height: float,
                 enable_drag: bool = True,
                 dv_range: float = 0.7,
                 da_range: float = 1.5,
                 goal_height: float = GOAL_TOP_HEIGHT_M,
                 goal_depth: float = GOAL_DEPTH_M,
                 wind: float = 0.0,
                 with_diagnostics: bool = False):
    """
    Robust make-window selector on the deployed command lattice.

    Each exact speed/angle lattice is integrated once into the target-
    independent event map X(v, angle). For a requested distance the map is
    thresholded into the same made/miss grid used by the season selector, and
    a two-dimensional prefix sum evaluates every rectangular perturbation box.
    The cache changes evaluation cost, not the decision rule.

    ``dv_range`` and ``da_range`` are bounded engineering half-widths.  The
    resulting fraction is a robustness score, not a calibrated probability.
    """
    values = [goal_distance, launch_height, dv_range, da_range,
              goal_height, goal_depth, wind]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("trajectory inputs must be finite numbers")
    goal_distance = float(goal_distance); launch_height = float(launch_height)
    dv_range = float(dv_range); da_range = float(da_range)
    if goal_distance <= 0.0 or launch_height < 0.0 or goal_depth <= 2.0 * ARTIFACT_R:
        raise ValueError("invalid launch or goal geometry")
    if not (0.0 < dv_range <= 2.0 and 0.0 < da_range <= 10.0):
        raise ValueError("robustness half-widths must satisfy 0<dv<=2 m/s and 0<da<=10 deg")

    # The exact event map is integrated in one vectorized batch and cached.
    # Prefix sums then evaluate every candidate window without re-integrating
    # the same trajectories for each command or target.
    v_fine = _expanded_grid(V_MIN_CMD, V_MAX_CMD, dv_range, 0.025)
    a_fine = _expanded_grid(A_MIN_CMD, A_MAX_CMD, da_range, 0.10)
    made_fine = _made_grid(
        v_fine, a_fine, launch_height, goal_height, goal_depth,
        goal_distance, wind, enable_drag)
    fine = _best_make_window_on_grid(
        v_fine, a_fine, made_fine, dv_range, da_range,
        V_MIN_CMD, V_MAX_CMD, A_MIN_CMD, A_MAX_CMD)
    if fine is None:
        return (None, {}) if with_diagnostics else None

    v_best = fine["velocity"]
    a_best = fine["angle_deg"]
    coverage = float(fine["fraction"])

    best = ShotParams(
        velocity=v_best, angle_deg=a_best,
        launch_height=launch_height, goal_distance=goal_distance,
        goal_height=goal_height, goal_depth=goal_depth,
        wind=wind, enable_drag=enable_drag,
    )
    if not with_diagnostics:
        return best

    # Auxiliary geometric margins are retained for UI diagnostics only.
    sur = get_surrogate(launch_height, goal_height, wind, enable_drag)
    entry_x_min, entry_x_max = _entry_x_bounds(goal_distance, goal_depth)
    a_b, v_lo, v_hi = band(sur, entry_x_min, entry_x_max)
    margin_norm = float(_margin_map(
        np.array([a_best]), np.array([v_best]), a_b, v_lo, v_hi,
        dv_range, da_range)[0])
    diag = {
        "window_pct":   100.0 * coverage,
        "coverage":     coverage,
        # Legacy aliases kept so older clients continue to render.
        "p_make":       coverage,
        "margin_sigma": margin_norm,
        "margin_norm":  margin_norm,
        "monotone_ok":  sur.monotone,
        "hits":         int(fine["hits"]),
        "trials":       int(fine["total"]),
    }
    diag.update(directional_margins(v_best, a_best, a_b, v_lo, v_hi))
    return best, diag


# ── Tolerance sweep ─────────────────────────────────────────────────────────
def _downsample(path: list, target: int = 80):
    # Rounded on the way out — 0.1 mm is far finer than a pixel, and full
    # float repr would triple the payload.
    _, xs, ys, _, _ = _densify(path, target)
    return [round(z, 4) for z in xs], [round(z, 4) for z in ys]


def make_sweep(p: ShotParams,
               dv_range: float = 0.7,
               da_range: float = 1.5,
               n: int = 11) -> dict:
    """
    Tolerance analysis around a nominal shot.

    The reported statistic is the deterministic made fraction on the same
    bounded perturbation lattice used by the selector. It is robust coverage,
    not a probability model. Only the trajectories needed for the drawing are
    retained as paths.
    """
    _validate_shot(p)
    sigma_v = float(dv_range)
    sigma_a = float(da_range)
    if not (math.isfinite(sigma_v) and math.isfinite(sigma_a)
            and 0.0 < sigma_v <= 2.0 and 0.0 < sigma_a <= 10.0):
        raise ValueError("invalid robustness half-width")

    sur = get_surrogate(p.launch_height, p.goal_height, p.wind, p.enable_drag)
    entry_x_min, entry_x_max = _entry_x_bounds(p.goal_distance, p.goal_depth)
    a_b, v_lo, v_hi = band(sur, entry_x_min, entry_x_max)

    # Coverage uses the selector lattice and is independent of the number of
    # representative paths requested for drawing.
    v_box = _uniform(p.velocity - sigma_v, p.velocity + sigma_v, 0.025)
    a_box = _uniform(p.angle_deg - sigma_a, p.angle_deg + sigma_a, 0.10)
    window = float(_made_grid(
        v_box, a_box, p.launch_height, p.goal_height, p.goal_depth,
        p.goal_distance, p.wind, p.enable_drag).mean())
    margins = directional_margins(p.velocity, p.angle_deg, a_b, v_lo, v_hi)
    m_sigma = float(_margin_map(np.array([p.angle_deg]), np.array([p.velocity]),
                                a_b, v_lo, v_hi, sigma_v, sigma_a)[0])

    # ── Which trajectories to draw ──────────────────────────────────────────
    ring = max(4, min(12, int(n) - 1))
    specs = [(0.0, 0.0, "nominal")]
    for scale in (1.0, 2.0):
        for i in range(ring):
            th = 2 * math.pi * i / ring
            specs.append((scale * sigma_v * math.cos(th),
                          scale * sigma_a * math.sin(th), "ring"))
    for k in (-2, -1, 0, 1, 2):
        a_edge = p.angle_deg + k * sigma_a
        for edge in (v_lo, v_hi):
            ve = float(np.interp(a_edge, a_b, edge, left=math.nan, right=math.nan))
            if not math.isnan(ve):
                specs.append((ve - p.velocity, k * sigma_a, "boundary"))

    vels = [max(0.1, p.velocity + dv) for dv, _, _ in specs]
    angs = [p.angle_deg + da        for _, da, _ in specs]

    r = _integrate(vels, angs, p.launch_height,
                   goal_height=p.goal_height, wind=p.wind, enable_drag=p.enable_drag,
                   dt=DT_SURROGATE, t_max=3.0, launch_x=p.launch_x,
                   front_x=p.goal_distance,
                   x_max=p.goal_distance + p.goal_depth + 2.0,
                   stop_at_rim=False, record_path=True)
    made = _made_from(r["x_at_top"], r["descending"], entry_x_min, entry_x_max)

    trajs = []
    for i, (dv, da, kind) in enumerate(specs):
        if kind == "nominal":
            continue
        xs, ys = _downsample(r["paths"][i])
        yf = r["y_at_front"][i]
        trajs.append({
            "x": xs, "y": ys,
            "made": bool(made[i]),
            "impact_y_at_goal": None if math.isnan(yf) else float(yf),
            "dv": dv, "da": da, "kind": kind,
        })

    nx, ny = _downsample(r["paths"][0])

    def _clean(arr):
        return [None if math.isnan(z) else round(float(z), 4) for z in arr]

    return {
        "trajectories": trajs,
        "nominal": {"x": nx, "y": ny, "made": bool(made[0])},
        "made_pct": 100.0 * window,
        "coverage": window,
        "p_make":   window,
        "window_pct": 100.0 * window,
        "margin_sigma": m_sigma,
        "margins":  margins,
        "band": {
            "angle_deg": [round(float(z), 2) for z in a_b],
            "v_lo": _clean(v_lo),
            "v_hi": _clean(v_hi),
        },
        "half_width": {"v": sigma_v, "a": sigma_a},
        "sigma": {"v": sigma_v, "a": sigma_a},
        "nominal_shot": {"velocity": p.velocity, "angle_deg": p.angle_deg},
        "goal": {
            "distance": p.goal_distance,
            "height":   p.goal_height,
            "depth":    p.goal_depth,
        },
    }


# ── Shot family: every commandable shot that scores at one distance ─────────
def shot_family(p: ShotParams,
                dv_range:   float = 0.7,
                da_range:   float = 1.5,
                angle_step: float = 1.0,
                trials:     int   = 21) -> dict:
    """
    One robust scoring representative per commandable launch angle.

    The cached exact event map is thresholded to made/miss and a prefix sum
    evaluates the same bounded make-window criterion as ``find_optimal``.
    ``trials`` remains for API compatibility; the calculation uses a fixed
    lattice so the number of curves drawn cannot change reported coverage.
    """
    _validate_shot(p)
    sigma_v = float(dv_range)
    sigma_a = float(da_range)
    if not (math.isfinite(sigma_v) and math.isfinite(sigma_a)
            and 0.0 < sigma_v <= 2.0 and 0.0 < sigma_a <= 10.0):
        raise ValueError("invalid robustness half-width")

    sur = get_surrogate(p.launch_height, p.goal_height, p.wind, p.enable_drag)
    entry_x_min, entry_x_max = _entry_x_bounds(p.goal_distance, p.goal_depth)
    entry_width = entry_x_max - entry_x_min
    a_b, v_lo, v_hi = band(sur, entry_x_min, entry_x_max)

    step = min(5.0, max(0.1, float(angle_step)))
    v_grid = _expanded_grid(V_MIN_CMD, V_MAX_CMD, sigma_v, 0.025)
    a_grid = _expanded_grid(A_MIN_CMD, A_MAX_CMD, sigma_a, 0.10)
    made_grid = _made_grid(
        v_grid, a_grid, p.launch_height, p.goal_height, p.goal_depth,
        p.goal_distance, p.wind, p.enable_drag)
    prefix = np.pad(made_grid.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    requested = _uniform(A_MIN_CMD, A_MAX_CMD, step)
    angle_indices = sorted(set(int(np.argmin(np.abs(a_grid - a))) for a in requested))
    velocity_indices = np.flatnonzero(
        (v_grid >= V_MIN_CMD - 1e-9) & (v_grid <= V_MAX_CMD + 1e-9)
        & (v_grid - sigma_v >= v_grid[0] - 1e-9)
        & (v_grid + sigma_v <= v_grid[-1] + 1e-9))

    selected = []
    for ia in angle_indices:
        a_value = float(a_grid[ia])
        if not (A_MIN_CMD - 1e-9 <= a_value <= A_MAX_CMD + 1e-9):
            continue
        a0 = int(np.searchsorted(a_grid, a_value - sigma_a, side="left"))
        a1 = int(np.searchsorted(a_grid, a_value + sigma_a, side="right"))
        best_at_angle = None
        for iv in velocity_indices:
            if not made_grid[iv, ia]:
                continue
            v0 = int(np.searchsorted(v_grid, v_grid[iv] - sigma_v, side="left"))
            v1 = int(np.searchsorted(v_grid, v_grid[iv] + sigma_v, side="right"))
            total = (v1 - v0) * (a1 - a0)
            hits = int(prefix[v1, a1] - prefix[v0, a1]
                       - prefix[v1, a0] + prefix[v0, a0])
            coverage = hits / total
            key = (coverage, hits, -float(v_grid[iv]))
            if best_at_angle is None or key > best_at_angle["key"]:
                best_at_angle = {
                    "key": key,
                    "velocity": float(v_grid[iv]),
                    "angle_deg": a_value,
                    "coverage": coverage,
                }
        if best_at_angle is not None:
            selected.append(best_at_angle)

    empty = {
        "shots": [], "best_index": None, "count": 0,
        "angle_span": None,
        "half_width": {"v": sigma_v, "a": sigma_a},
        "sigma": {"v": sigma_v, "a": sigma_a},
        "goal": {"distance": p.goal_distance, "height": p.goal_height,
                 "depth": p.goal_depth},
        "launch_height": p.launch_height,
        "angle_step": step,
    }
    if not selected:
        return empty

    a_f = np.array([row["angle_deg"] for row in selected])
    v_best = np.array([row["velocity"] for row in selected])
    w_best = np.array([row["coverage"] for row in selected])
    lo_f = np.clip(np.interp(a_f, a_b, v_lo), V_MIN_CMD, V_MAX_CMD)
    hi_f = np.clip(np.interp(a_f, a_b, v_hi), V_MIN_CMD, V_MAX_CMD)
    m_best = _margin_map(a_f, v_best, a_b, v_lo, v_hi, sigma_v, sigma_a)

    top   = np.flatnonzero(w_best >= w_best.max() - 1e-12)
    i_top = int(top[np.argmin(v_best[top])])

    r = _integrate(v_best, a_f, p.launch_height,
                   goal_height=p.goal_height, wind=p.wind, enable_drag=p.enable_drag,
                   dt=DT_SURROGATE, t_max=3.0, launch_x=p.launch_x,
                   front_x=p.goal_distance,
                   x_max=p.goal_distance + p.goal_depth + 2.0,
                   stop_at_rim=False, record_path=True)
    made = _made_from(r["x_at_top"], r["descending"], entry_x_min, entry_x_max)

    shots = []
    for i in range(a_f.size):
        xs, ys = _downsample(r["paths"][i], 70)
        xt = r["x_at_top"][i]
        shots.append({
            "angle_deg":  round(float(a_f[i]), 2),
            "velocity":   round(float(v_best[i]), 4),
            "v_lo":       round(float(lo_f[i]), 4),
            "v_hi":       round(float(hi_f[i]), 4),
            "window_pct": round(100.0 * float(w_best[i]), 3),
            "margin_sigma": round(float(m_best[i]), 4),
            "coverage":   float(w_best[i]),
            "p_make":     float(w_best[i]),
            # 0 = front lip, 1 = back lip.  NaN if it never crosses the rim.
            "entry_pos":  None if math.isnan(xt) else round(
                              float((xt - entry_x_min) / entry_width), 4),
            "entry_angle_deg": None if math.isnan(r["entry_angle_deg"][i]) else round(
                              float(r["entry_angle_deg"][i]), 2),
            "flight_time": None if math.isnan(r["t_at_top"][i]) else round(
                              float(r["t_at_top"][i]), 4),
            "apex_y":      round(float(r["apex_y"][i]), 4),
            "made":        bool(made[i]),
            "x": xs, "y": ys,
        })

    return {
        **empty,
        "shots":       shots,
        "count":       len(shots),
        "best_index":  i_top,
        "angle_span":  [round(float(a_f[0]), 2), round(float(a_f[-1]), 2)],
    }


# ── Monte-Carlo cloud: what the shooter's own scatter actually looks like ───
def monte_carlo(p: ShotParams,
                dv_range: float = 0.7,
                da_range: float = 1.5,
                n:        int   = 250,
                seed:     Optional[int] = None,
                path_points: int = 45) -> dict:
    """
    N shots sampled uniformly from the declared engineering perturbation box.

    This is a visual sensitivity cloud.  It deliberately uses the same bounded
    uncertainty semantics as the selector and does not imply a fitted noise
    distribution.
    """
    _validate_shot(p)
    sigma_v = float(dv_range)
    sigma_a = float(da_range)
    if not (math.isfinite(sigma_v) and math.isfinite(sigma_a)
            and 0.0 < sigma_v <= 2.0 and 0.0 < sigma_a <= 10.0):
        raise ValueError("invalid robustness half-width")
    n = max(1, min(600, int(n)))

    entry_x_min, entry_x_max = _entry_x_bounds(p.goal_distance, p.goal_depth)

    rng  = np.random.default_rng(seed)
    dv   = rng.uniform(-sigma_v, sigma_v, n)
    da   = rng.uniform(-sigma_a, sigma_a, n)
    vels = np.maximum(p.velocity + dv, 0.1)
    angs = p.angle_deg + da

    r = _integrate(vels, angs, p.launch_height,
                   goal_height=p.goal_height, wind=p.wind, enable_drag=p.enable_drag,
                   dt=DT_SURROGATE, t_max=3.0, launch_x=p.launch_x,
                   front_x=p.goal_distance,
                   x_max=p.goal_distance + p.goal_depth + 2.0,
                   stop_at_rim=False, record_path=True)
    made = _made_from(r["x_at_top"], r["descending"], entry_x_min, entry_x_max)

    samples = []
    for i in range(n):
        _, xs, ys, _, _ = _densify(r["paths"][i], path_points)
        samples.append({
            "x": [round(z, 3) for z in xs],
            "y": [round(z, 3) for z in ys],
            "made": bool(made[i]),
            "velocity": round(float(vels[i]), 3),
            "angle_deg": round(float(angs[i]), 2),
        })

    hits = int(made.sum())
    reference = make_sweep(p, dv_range=sigma_v, da_range=sigma_a, n=21)["coverage"]
    return {
        "samples":  samples,
        "n":        n,
        "hits":     hits,
        "p_empirical": hits / n,
        # Binomial standard error, so the UI can say how tight the estimate is.
        "p_stderr": math.sqrt(max(hits / n * (1.0 - hits / n), 0.0) / n),
        "p_analytic": reference,
        "coverage_reference": reference,
        "distribution": "uniform_box",
        "half_width": {"v": sigma_v, "a": sigma_a},
        "sigma":    {"v": sigma_v, "a": sigma_a},
        "nominal_shot": {"velocity": p.velocity, "angle_deg": p.angle_deg},
        "goal": {"distance": p.goal_distance, "height": p.goal_height,
                 "depth": p.goal_depth},
        "launch_height": p.launch_height,
    }


# ── Batch solve: the most robust shot at each of many distances ─────────────
def sweep_targets(distances,
                  launch_height: float,
                  enable_drag: bool = True,
                  dv_range: float = 0.7,
                  da_range: float = 1.5,
                  goal_height: float = GOAL_TOP_HEIGHT_M,
                  goal_depth:  float = GOAL_DEPTH_M,
                  wind: float = 0.0) -> List[dict]:
    """
    `find_optimal` once per distance — so every row is picked by make-window
    fraction.  The surrogate does not depend on goal distance, so every
    distance after the first is pure array arithmetic on an already-built
    cache.

    Unreachable distances come back as rows with `ok: False` rather than being
    dropped, so a caller plotting the result can show the gap where it is.
    """
    ds = [float(d) for d in distances]
    rows: List[dict] = []
    solved: List[Tuple[int, ShotParams]] = []

    for d in ds:
        row = {"distance_m": round(d, 5), "ok": False}
        if math.isfinite(d) and d > 0:
            best, diag = find_optimal(d, launch_height, enable_drag=enable_drag,
                                      dv_range=dv_range, da_range=da_range,
                                      goal_height=goal_height, goal_depth=goal_depth,
                                      wind=wind, with_diagnostics=True)
            if best is not None:
                row.update({
                    "ok":           True,
                    "velocity":     round(best.velocity, 4),
                    "angle_deg":    round(best.angle_deg, 3),
                    "window_pct":   round(diag["window_pct"], 3),
                    "margin_sigma": round(diag["margin_sigma"], 4),
                    "p_make":       diag["p_make"],
                    "v_down_ms":    diag["v_down_ms"],
                    "v_up_ms":      diag["v_up_ms"],
                    "a_down_deg":   diag["a_down_deg"],
                    "a_up_deg":     diag["a_up_deg"],
                })
                solved.append((len(rows), best))
        rows.append(row)

    # One batched integration for the flight telemetry of every winner.
    if solved:
        r = _integrate([b.velocity for _, b in solved],
                       [b.angle_deg for _, b in solved],
                       launch_height, goal_height=goal_height, wind=wind,
                       enable_drag=enable_drag, dt=DT_SURROGATE, stop_at_rim=True)
        for k, (idx, best) in enumerate(solved):
            lo_x, hi_x = _entry_x_bounds(best.goal_distance, goal_depth)
            xt = r["x_at_top"][k]
            rows[idx].update({
                "entry_angle_deg": None if math.isnan(r["entry_angle_deg"][k]) else round(
                                       float(r["entry_angle_deg"][k]), 2),
                "flight_time":     None if math.isnan(r["t_at_top"][k]) else round(
                                       float(r["t_at_top"][k]), 4),
                "entry_pos":       None if math.isnan(xt) else round(
                                       float((xt - lo_x) / (hi_x - lo_x)), 4),
                "apex_y":          round(float(r["apex_y"][k]), 4),
            })
    return rows


# ── Lookup table ─────────────────────────────────────────────────────────────
def build_lut(p: ShotParams,
              v_min: float = 2.0,  v_max: float = 12.0,  v_steps: int = 50,
              a_min: float = 20.0, a_max: float = 80.0,  a_steps: int = 61,
              ) -> List[dict]:
    _validate_shot(p)
    if not (math.isfinite(v_min) and math.isfinite(v_max)
            and math.isfinite(a_min) and math.isfinite(a_max)
            and 1 <= int(v_steps) <= 500 and 1 <= int(a_steps) <= 500
            and 0.0 < v_min <= v_max and -89.9 < a_min <= a_max < 89.9):
        raise ValueError("invalid lookup-table bounds")
    vs = np.linspace(v_min, v_max, int(v_steps))
    as_ = np.linspace(a_min, a_max, int(a_steps))
    VV, AA = np.meshgrid(vs, as_, indexing="ij")

    r = _integrate(VV.ravel(), AA.ravel(), p.launch_height,
                   goal_height=p.goal_height, wind=p.wind, enable_drag=p.enable_drag,
                   dt=DT_PATH, t_max=4.0, launch_x=p.launch_x,
                   front_x=p.goal_distance,
                   x_max=p.goal_distance + p.goal_depth + 2.0,
                   stop_at_rim=False)
    entry_x_min, entry_x_max = _entry_x_bounds(p.goal_distance, p.goal_depth)
    made = _made_from(r["x_at_top"], r["descending"], entry_x_min, entry_x_max)

    def _r(val, nd):
        return "" if math.isnan(val) else round(float(val), nd)

    rows = []
    for k in range(VV.size):
        rows.append({
            "velocity_ms":      round(float(VV.ravel()[k]), 3),
            "angle_deg":        round(float(AA.ravel()[k]), 2),
            "made":             int(made[k]),
            "x_at_top_m":       _r(r["x_at_top"][k], 4),
            "impact_y_front_m": _r(r["y_at_front"][k], 4),
            "entry_angle_deg":  _r(r["entry_angle_deg"][k], 2),
            "flight_time_s":    _r(r["t_at_top"][k], 4),
            "apex_y_m":         round(float(r["apex_y"][k]), 4),
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
    best, diag = find_optimal(in_to_m(60), in_to_m(15.75), with_diagnostics=True)
    if best:
        rb = simulate(best)
        print(f"\noptimal: v={best.velocity:.2f} m/s  angle={best.angle_deg:.1f}°  made={rb.made}")
        print(f"         window={diag['window_pct']:.1f}%  margin={diag['margin_sigma']:.2f}σ  "
              f"P(make)={diag['p_make']*100:.1f}%")
        print(f"         slack: -{diag['v_down_ms']:.2f}/+{diag['v_up_ms']:.2f} m/s, "
              f"-{diag['a_down_deg']:.1f}/+{diag['a_up_deg']:.1f}°")
