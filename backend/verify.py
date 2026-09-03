"""Verification and benchmark harness for the DECODE trajectory simulator.

Run from this directory with ``python verify.py``. The suite checks the
vacuum analytic solution, an optional independent SciPy/DOP853 reference,
event location, the cached event map, the robust make-window decision rule,
public API consistency, input validation, and cache concurrency.
"""

import math
import threading
import time

import numpy as np

import physics as P

try:
    from scipy.integrate import solve_ivp
except ImportError:  # Runtime users need NumPy only; SciPy is a validation extra.
    solve_ivp = None


FAILS = []
SKIPS = []
H = P.in_to_m(15.75)
DISTANCES = [P.in_to_m(value) for value in (60, 70, 80, 90, 100, 105)]


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def skip(name: str, detail: str) -> None:
    print(f"  SKIP  {name}   {detail}")
    SKIPS.append(name)


def test_constant_drag() -> None:
    speeds = np.geomspace(0.1, 100.0, 500)
    got = P._cd_np(speeds)
    check("constant effective drag coefficient",
          np.all(got == P.CD_EFFECTIVE)
          and all(P._cd(float(v)) == P.CD_EFFECTIVE for v in speeds),
          f"Cd={P.CD_EFFECTIVE:.2f}")


def test_vacuum_analytic() -> None:
    velocities = np.linspace(P.V_MIN_CMD, P.V_MAX_CMD, 31)
    angles = np.linspace(P.A_MIN_CMD, P.A_MAX_CMD, 33)
    vv, aa = np.meshgrid(velocities, angles, indexing="ij")
    result = P._integrate(vv.ravel(), aa.ravel(), H, enable_drag=False,
                          dt=P.DT_SURROGATE, stop_at_rim=True)

    theta = np.radians(aa.ravel())
    vx0 = vv.ravel() * np.cos(theta)
    vy0 = vv.ravel() * np.sin(theta)
    discriminant = vy0 * vy0 + 2.0 * P.G * (H - P.GOAL_TOP_HEIGHT_M)
    reachable = discriminant >= 0.0
    t_cross = np.full(vx0.size, np.nan)
    t_cross[reachable] = (vy0[reachable] + np.sqrt(discriminant[reachable])) / P.G
    x_exact = vx0 * t_cross

    valid = reachable & np.isfinite(result["x_at_top"])
    missing = int(np.count_nonzero(reachable != np.isfinite(result["x_at_top"])))
    max_x = float(np.max(np.abs(result["x_at_top"][valid] - x_exact[valid])))
    max_t = float(np.max(np.abs(result["t_at_top"][valid] - t_cross[valid])))
    check("vacuum rim event matches analytic projectile motion",
          missing == 0 and max_x < 2e-6 and max_t < 4e-7,
          f"{valid.sum()} cases; max |dx|={max_x:.2e} m, |dt|={max_t:.2e} s")


def _adaptive_reference(velocity: float, angle_deg: float, launch_height: float):
    theta = math.radians(angle_deg)
    y0 = [0.0, launch_height,
          velocity * math.cos(theta), velocity * math.sin(theta)]
    k_drag = 0.5 * P.RHO_AIR * P.CD_EFFECTIVE * P.ARTIFACT_A / P.ARTIFACT_M

    def rhs(_t, state):
        _, _, vx, vz = state
        speed = math.hypot(vx, vz)
        return [vx, vz, -k_drag * speed * vx, -P.G - k_drag * speed * vz]

    def rim_event(_t, state):
        return state[1] - P.GOAL_TOP_HEIGHT_M

    rim_event.direction = -1
    rim_event.terminal = True
    solution = solve_ivp(
        rhs, (0.0, 3.0), y0, method="DOP853",
        rtol=1e-11, atol=1e-13, max_step=0.002, events=rim_event,
    )
    if not solution.t_events[0].size:
        return math.nan, math.nan
    return float(solution.y_events[0][0, 0]), float(solution.t_events[0][0])


def test_independent_adaptive_solver() -> None:
    if solve_ivp is None:
        skip("independent DOP853 comparison", "install scipy to enable")
        return
    cases = [(v, a) for v in np.linspace(P.V_MIN_CMD, P.V_MAX_CMD, 13)
             for a in np.linspace(P.A_MIN_CMD, P.A_MAX_CMD, 13)]
    fast = P._integrate([v for v, _ in cases], [a for _, a in cases], H,
                        dt=P.DT_SURROGATE, stop_at_rim=True)
    worst_x = worst_t = 0.0
    mismatch = 0
    compared = 0
    for index, (velocity, angle) in enumerate(cases):
        x_ref, t_ref = _adaptive_reference(velocity, angle, H)
        x_fast = float(fast["x_at_top"][index])
        t_fast = float(fast["t_at_top"][index])
        if math.isnan(x_ref) != math.isnan(x_fast):
            mismatch += 1
        elif not math.isnan(x_ref):
            compared += 1
            worst_x = max(worst_x, abs(x_fast - x_ref))
            worst_t = max(worst_t, abs(t_fast - t_ref))
    check("custom solver agrees with independent SciPy DOP853",
          mismatch == 0 and worst_x < 1e-4 and worst_t < 2e-5,
          f"{compared} events; max |dx|={worst_x:.2e} m, |dt|={worst_t:.2e} s")


def test_event_convergence() -> None:
    velocities = np.linspace(3.5, 8.0, 37)
    angles = np.linspace(30.0, 70.0, 41)
    vv, aa = np.meshgrid(velocities, angles, indexing="ij")
    fast = P._integrate(vv.ravel(), aa.ravel(), H, dt=P.DT_SURROGATE, stop_at_rim=True)
    ref = P._integrate(vv.ravel(), aa.ravel(), H, dt=0.001, stop_at_rim=True)
    valid = np.isfinite(fast["x_at_top"]) & np.isfinite(ref["x_at_top"])
    mismatch = int(np.count_nonzero(np.isfinite(fast["x_at_top"])
                                    != np.isfinite(ref["x_at_top"])))
    max_error = float(np.max(np.abs(fast["x_at_top"][valid] - ref["x_at_top"][valid])))
    check("20 ms event solution converges to 1 ms self-reference",
          mismatch == 0 and max_error < 1e-4,
          f"{valid.sum()} events; max |dx|={max_error:.2e} m")


def test_hermite_bisection() -> None:
    p0 = np.array([-1.2, -0.4, -0.3])
    p1 = np.array([0.8, 1.1, 1.7])
    d0 = np.array([1.0, 0.7, 0.4])
    d1 = np.array([1.2, 0.9, 0.6])
    target = np.zeros(3)
    tau = P._hermite_root(p0, d0, p1, d1, 0.02, target)
    residual = np.max(np.abs(P._hermite_eval(p0, d0, p1, d1, 0.02, tau) - target))
    check("Hermite event root uses fixed bracketed bisection",
          np.all((tau >= 0.0) & (tau <= 1.0)) and residual < 4e-5,
          f"16 halvings; normalized bracket <= {2.0 ** -16:.2e}; residual={residual:.2e}")


def test_event_map_cache() -> None:
    P.clear_surrogate_cache()
    v_grid = P._expanded_grid(P.V_MIN_CMD, P.V_MAX_CMD, 0.7, 0.10)
    a_grid = P._expanded_grid(P.A_MIN_CMD, P.A_MAX_CMD, 1.5, 0.25)
    first = P._event_grid(v_grid, a_grid, H, P.GOAL_TOP_HEIGHT_M, 0.0, True)
    second = P._event_grid(v_grid, a_grid, H, P.GOAL_TOP_HEIGHT_M, 0.0, True)
    vv, aa = np.meshgrid(v_grid, a_grid, indexing="ij")
    direct = P._integrate(vv.ravel(), aa.ravel(), H,
                          dt=P.DT_SURROGATE, stop_at_rim=True)
    same = np.allclose(np.nan_to_num(first[0], nan=-1.0),
                       np.nan_to_num(direct["x_at_top"].reshape(first[0].shape), nan=-1.0))
    check("cached X(v,theta) event map equals direct batch integration", same)
    check("event-map cache returns the same tile", first is second)


def test_selector_rule() -> None:
    violations = 0
    max_delta = 0.0
    for distance in DISTANCES:
        best, diag = P.find_optimal(distance, H, dv_range=0.7, da_range=1.5,
                                    with_diagnostics=True)
        if best is None or not P.simulate(best).made:
            violations += 1
            continue
        if not (P.V_MIN_CMD <= best.velocity <= P.V_MAX_CMD
                and P.A_MIN_CMD <= best.angle_deg <= P.A_MAX_CMD):
            violations += 1

        v_grid = P._expanded_grid(P.V_MIN_CMD, P.V_MAX_CMD, 0.7, 0.025)
        a_grid = P._expanded_grid(P.A_MIN_CMD, P.A_MAX_CMD, 1.5, 0.10)
        made = P._made_grid(v_grid, a_grid, H, P.GOAL_TOP_HEIGHT_M,
                            P.GOAL_DEPTH_M, distance, 0.0, True)
        exhaustive = P._best_make_window_on_grid(
            v_grid, a_grid, made, 0.7, 1.5,
            P.V_MIN_CMD, P.V_MAX_CMD, P.A_MIN_CMD, P.A_MAX_CMD)
        if exhaustive is None:
            violations += 1
            continue
        max_delta = max(max_delta, abs(diag["coverage"] - exhaustive["fraction"]))
        if (best.velocity != exhaustive["velocity"]
                or best.angle_deg != exhaustive["angle_deg"]
                or diag["hits"] != exhaustive["hits"]
                or diag["trials"] != exhaustive["total"]):
            violations += 1
    check("selector returns the exhaustive maximum make-window command",
          violations == 0 and max_delta == 0.0,
          f"{len(DISTANCES)} distances; violations={violations}; max delta={max_delta:.2e}")


def test_sweep_semantics() -> None:
    best = P.find_optimal(DISTANCES[1], H, dv_range=0.7, da_range=1.5)
    values = [P.make_sweep(best, 0.7, 1.5, n=n)["coverage"] for n in (5, 11, 21)]
    check("robust coverage is independent of drawing resolution",
          max(values) - min(values) < 1e-15, f"coverage={values[0]:.6f}")

    sample = P.monte_carlo(best, 0.7, 1.5, n=5000, seed=17)
    reference = sample["coverage_reference"]
    se = math.sqrt(max(reference * (1.0 - reference), 1e-9) / sample["n"])
    z = abs(sample["p_empirical"] - reference) / se
    check("uniform-box sample agrees with deterministic grid coverage",
          sample["distribution"] == "uniform_box" and z < 4.0,
          f"sample={sample['p_empirical']:.4f}, grid={reference:.4f}, z={z:.2f}")


def test_public_outputs() -> None:
    distance = DISTANCES[2]
    best = P.find_optimal(distance, H, dv_range=0.7, da_range=1.5)
    result = P.simulate(best)
    lut = P.build_lut(best, v_min=best.velocity, v_max=best.velocity, v_steps=1,
                      a_min=best.angle_deg, a_max=best.angle_deg, a_steps=1)[0]
    check("public flight time is the rim event time",
          result.t_at_top is not None
          and abs(lut["flight_time_s"] - result.t_at_top) < 6e-5
          and result.t_at_top < result.t[-1],
          f"rim={result.t_at_top:.4f} s, path end={result.t[-1]:.4f} s")

    family = P.shot_family(best, 0.7, 1.5, angle_step=2.0)
    all_score = family["count"] > 0 and all(P.simulate(P.ShotParams(
        velocity=row["velocity"], angle_deg=row["angle_deg"], launch_height=H,
        goal_distance=distance)).made for row in family["shots"])
    check("every advertised shot-family command scores", all_score,
          f"{family['count']} commands")


def test_validation() -> None:
    cases = [
        lambda: P.simulate(P.ShotParams(math.nan, 50.0, H)),
        lambda: P.simulate(P.ShotParams(5.0, 90.0, H)),
        lambda: P.find_optimal(1.5, H, dv_range=0.0),
        lambda: P.make_sweep(P.ShotParams(5.0, 50.0, H), da_range=math.inf),
        lambda: P._integrate([5.0], [50.0, 51.0], H),
    ]
    rejected = 0
    for case in cases:
        try:
            case()
        except ValueError:
            rejected += 1
    check("invalid public inputs fail fast", rejected == len(cases),
          f"{rejected}/{len(cases)} rejected")


def test_cache_concurrency() -> None:
    P.clear_surrogate_cache()
    errors = []
    results = []

    def worker(index):
        try:
            results.append(P.find_optimal(DISTANCES[index % len(DISTANCES)], H,
                                          dv_range=0.7, da_range=1.5))
        except Exception as exc:  # noqa: BLE001 - test records all worker failures.
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    check("concurrent event-map evaluation is safe",
          not errors and len(results) == len(threads), f"errors={errors}")
    check("event-map cache respects its size cap",
          len(P._EVENT_CACHE) <= P._EVENT_MAX, f"size={len(P._EVENT_CACHE)}")


def benchmark() -> None:
    print("\n── implementation timings (median of five warm calls) ──")
    distance = DISTANCES[0]
    best = P.find_optimal(distance, H, dv_range=0.7, da_range=1.5)

    def median_ms(function, count=5):
        values = []
        for _ in range(count):
            start = time.perf_counter()
            function()
            values.append(1000.0 * (time.perf_counter() - start))
        return sorted(values)[len(values) // 2]

    rows = [
        ("find_optimal (cached X map)", lambda: P.find_optimal(distance, H, True, 0.7, 1.5)),
        ("make_sweep", lambda: P.make_sweep(best, 0.7, 1.5, 11)),
        ("simulate", lambda: P.simulate(best)),
        ("shot_family (2 deg)", lambda: P.shot_family(best, 0.7, 1.5, 2.0)),
    ]
    for label, function in rows:
        print(f"  {label:30s} {median_ms(function):8.2f} ms")


if __name__ == "__main__":
    print("── numerical and decision-rule verification ──")
    test_constant_drag()
    test_vacuum_analytic()
    test_independent_adaptive_solver()
    test_event_convergence()
    test_hermite_bisection()
    test_event_map_cache()
    test_selector_rule()
    test_sweep_semantics()
    test_public_outputs()
    test_validation()
    test_cache_concurrency()
    benchmark()
    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
    if SKIPS:
        print(f"OPTIONAL CHECKS SKIPPED: {SKIPS}")
    raise SystemExit(1 if FAILS else 0)
