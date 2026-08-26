"""
Verification + benchmark harness for the DECODE physics engine.

Run:  python verify.py

Checks the invariants the fast path depends on, then times the public entry
points.  Everything is checked against a high-resolution self-reference
(dt = 0.001) rather than against a fixed expected value, so the suite stays
meaningful if the physical constants are re-calibrated.
"""

import math
import threading
import time

import numpy as np

import physics as P


FAILS = []


def check(name: str, ok: bool, detail: str = ""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


CONFIGS = [
    # (launch_height_in, goal_distance_in, drag, wind)
    (15.75,  60.0, True,   0.0),
    (10.00,  40.0, True,   0.0),
    (20.00, 100.0, True,   0.0),
    (15.75,  24.0, True,   0.0),
    (15.75, 144.0, True,   0.0),
    (15.75,  60.0, False,  0.0),
    (15.75,  60.0, True,  +2.0),
    (15.75,  60.0, True,  -2.0),
]


# ── 1. Scalar vs vectorised drag coefficient ────────────────────────────────
def test_cd():
    v = np.exp(np.linspace(math.log(1e-4), math.log(400.0), 20000))
    got = P._cd_np(v)
    want = np.array([P._cd(float(z)) for z in v])
    # _cd_np floors Re at 1e-9; _cd floors at 1e-6. They only differ in the
    # Stokes regime far below any real shot speed.
    m = v > 1e-3
    err = np.abs(got[m] - want[m]).max()
    check("_cd_np matches scalar _cd", err < 1e-12, f"max|Δ|={err:.2e}")


# ── 2. Event location converges ─────────────────────────────────────────────
def test_event_convergence():
    """
    Two separate claims, because they have different error models.

    The rim crossing solves y(τ) = rim, so the x it reports carries
    dx = dy / tan(entry_angle).  A shot that grazes the rim nearly
    horizontally is genuinely ill-conditioned in x — that is geometry, not
    integration error — so x accuracy is only asserted where the crossing is
    well-conditioned.  What must hold everywhere is the made/miss DECISION,
    which is what the tool actually reports.
    """
    worst_cond = 0.0
    disagree = total = 0
    for h_in, d_in, drag, wind in CONFIGS:
        h, d = P.in_to_m(h_in), P.in_to_m(d_in)
        VV, AA = np.meshgrid(np.arange(2.0, 12.001, 0.2),
                             np.arange(20.0, 80.001, 0.75), indexing="ij")
        ref = P._integrate(VV.ravel(), AA.ravel(), h, wind=wind,
                           enable_drag=drag, dt=0.001, stop_at_rim=True)
        fast = P._integrate(VV.ravel(), AA.ravel(), h, wind=wind,
                            enable_drag=drag, dt=P.DT_SURROGATE, stop_at_rim=True)

        lo, hi = P._entry_x_bounds(d, P.GOAL_DEPTH_M)
        m_ref = P._made_from(ref["x_at_top"], ref["descending"], lo, hi)
        m_fast = P._made_from(fast["x_at_top"], fast["descending"], lo, hi)
        disagree += int((m_ref != m_fast).sum())
        total += m_ref.size

        with np.errstate(invalid="ignore"):
            good = (~np.isnan(ref["x_at_top"]) & ~np.isnan(fast["x_at_top"])
                    & (ref["entry_angle_deg"] > 5.0))
        if good.any():
            worst_cond = max(worst_cond, float(
                np.abs(ref["x_at_top"][good] - fast["x_at_top"][good]).max()))

    check("made/miss decision identical at dt=0.02 and dt=0.001",
          disagree == 0, f"{disagree}/{total} cells disagree")
    check("x_at_top matches dt=0.001 where the crossing is well-conditioned",
          worst_cond < 1e-4, f"max|Δx|={worst_cond:.2e} m for entry angle > 5°")


# ── 3. Monotonicity + single-interval invariants ────────────────────────────
def test_monotone():
    bad = []
    for h_in, d_in, drag, wind in CONFIGS:
        sur = P.get_surrogate(P.in_to_m(h_in), wind=wind, enable_drag=drag)
        if not sur.monotone:
            bad.append((h_in, drag, wind))
    check("x_at_top is monotone in v for every angle", not bad, f"violations: {bad}")

    # The made-set at each angle must be one contiguous interval.
    multi = 0
    for h_in, d_in, drag, wind in CONFIGS:
        sur = P.get_surrogate(P.in_to_m(h_in), wind=wind, enable_drag=drag)
        lo, hi = P._entry_x_bounds(P.in_to_m(d_in), P.GOAL_DEPTH_M)
        for j in range(sur.a_grid.size):
            col = sur.x_at_top[:, j]
            with np.errstate(invalid="ignore"):
                m = (col >= lo) & (col <= hi)
            idx = np.flatnonzero(m)
            if idx.size and idx[-1] - idx[0] + 1 != idx.size:
                multi += 1
    check("made-set per angle is a single interval", multi == 0, f"{multi} angles with gaps")


# ── 4. Band boundaries vs exact bisection ───────────────────────────────────
def test_band():
    worst = 0.0
    n = 0
    for h_in, d_in, drag, wind in CONFIGS[:5]:
        h, d = P.in_to_m(h_in), P.in_to_m(d_in)
        sur = P.get_surrogate(h, wind=wind, enable_drag=drag)
        lo_x, hi_x = P._entry_x_bounds(d, P.GOAL_DEPTH_M)
        a_b, v_lo, v_hi = P.band(sur, lo_x, hi_x)

        def xtop(v, a):
            r = P._integrate([v], [a], h, wind=wind, enable_drag=drag,
                             dt=0.001, stop_at_rim=True)
            return float(r["x_at_top"][0])

        for a in a_b[::120]:
            for interp_v, target in ((np.interp(a, a_b, v_lo), lo_x),
                                     (np.interp(a, a_b, v_hi), hi_x)):
                if math.isnan(interp_v):
                    continue
                a_lo, a_hi = max(0.5, interp_v - 0.4), interp_v + 0.4
                f_lo, f_hi = xtop(a_lo, a), xtop(a_hi, a)
                if math.isnan(f_lo) or math.isnan(f_hi) or not (f_lo <= target <= f_hi):
                    continue
                for _ in range(40):
                    mid = 0.5 * (a_lo + a_hi)
                    fm = xtop(mid, a)
                    if math.isnan(fm) or fm < target: a_lo = mid
                    else:                             a_hi = mid
                worst = max(worst, abs(interp_v - 0.5 * (a_lo + a_hi)))
                n += 1
    check("band boundaries match exact bisection", worst < 5e-3,
          f"max|Δv|={worst:.5f} m/s over {n} boundaries")


# ── 5. Analytic P(make) vs Monte Carlo ──────────────────────────────────────
def test_p_make():
    rng = np.random.default_rng(7)
    worst = 0.0
    for h_in, d_in, drag, wind in CONFIGS[:4]:
        h, d = P.in_to_m(h_in), P.in_to_m(d_in)
        sur = P.get_surrogate(h, wind=wind, enable_drag=drag)
        lo_x, hi_x = P._entry_x_bounds(d, P.GOAL_DEPTH_M)
        a_b, v_lo, v_hi = P.band(sur, lo_x, hi_x)
        best = P.find_optimal(d, h, enable_drag=drag, wind=wind,
                              dv_range=0.25, da_range=1.0)
        if best is None:
            continue
        sv, sa = 0.25, 1.0
        analytic = P.p_make_gaussian(best.velocity, best.angle_deg,
                                     a_b, v_lo, v_hi, sv, sa)
        N = 120_000
        vs = best.velocity + rng.normal(0, sv, N)
        as_ = best.angle_deg + rng.normal(0, sa, N)
        r = P._integrate(vs, as_, h, wind=wind, enable_drag=drag,
                         dt=P.DT_SURROGATE, stop_at_rim=True)
        mc = float(P._made_from(r["x_at_top"], r["descending"], lo_x, hi_x).mean())
        worst = max(worst, abs(analytic - mc))
    # 3 sigma of a 120k-sample binomial is ~0.005 at p=0.5
    check("analytic P(make) matches Monte Carlo", worst < 0.01,
          f"max|Δp|={worst:.4f} (120k samples, MC 3σ ≈ 0.004)")


# ── 6. Cross-path consistency ───────────────────────────────────────────────
def test_cross_path():
    h, d = P.in_to_m(15.75), P.in_to_m(60)
    sur = P.get_surrogate(h)
    lo_x, hi_x = P._entry_x_bounds(d, P.GOAL_DEPTH_M)
    rng = np.random.default_rng(3)
    vs = rng.uniform(2.5, 11.0, 300)
    as_ = rng.uniform(22.0, 79.0, 300)
    fast = P._integrate(vs, as_, h, dt=P.DT_SURROGATE, stop_at_rim=True)
    fast_made = P._made_from(fast["x_at_top"], fast["descending"], lo_x, hi_x)

    mism = 0
    worst = 0.0
    for k in range(vs.size):
        s = P.simulate(P.ShotParams(velocity=float(vs[k]), angle_deg=float(as_[k]),
                                    launch_height=h, goal_distance=d))
        if s.made != bool(fast_made[k]):
            mism += 1
        if s.x_at_top is not None and not np.isnan(fast["x_at_top"][k]):
            worst = max(worst, abs(s.x_at_top - fast["x_at_top"][k]))
    check("simulate() and the surrogate agree on made", mism == 0, f"{mism}/300 mismatches")
    check("simulate() and the surrogate agree on x_at_top", worst < 1e-6,
          f"max|Δx|={worst:.2e} m")


# ── 7. Optimiser actually maximises robustness ──────────────────────────────
def test_optimum():
    worse = 0
    for h_in, d_in, drag, wind in CONFIGS[:5]:
        h, d = P.in_to_m(h_in), P.in_to_m(d_in)
        sv, sa = 0.25, 1.0
        best, diag = P.find_optimal(d, h, enable_drag=drag, wind=wind,
                                    dv_range=sv, da_range=sa, with_diagnostics=True)
        if best is None:
            continue
        sur = P.get_surrogate(h, wind=wind, enable_drag=drag)
        lo_x, hi_x = P._entry_x_bounds(d, P.GOAL_DEPTH_M)
        a_b, v_lo, v_hi = P.band(sur, lo_x, hi_x)
        # random probe: nothing should beat the reported margin
        rng = np.random.default_rng(11)
        pv = rng.uniform(P.V_MIN_CMD, P.V_MAX_CMD, 4000)
        pa = rng.uniform(P.A_MIN_CMD, P.A_MAX_CMD, 4000)
        m = P._margin_map(pa, pv, a_b, v_lo, v_hi, sv, sa)
        if float(m.max()) > diag["margin_sigma"] + 5e-3:
            worse += 1
        # the returned shot must actually score
        r = P.simulate(best)
        if not r.made:
            worse += 1
    check("optimum is a true maximum and scores", worse == 0, f"{worse} violations")


# ── 8. Cache determinism and thread safety ──────────────────────────────────
def test_cache():
    P.clear_surrogate_cache()
    h = P.in_to_m(15.75)
    a = P.get_surrogate(h)
    b = P.get_surrogate(h)
    check("cache returns the same object on a hit", a is b)

    P.clear_surrogate_cache()
    c = P.build_surrogate(h)
    same = np.allclose(np.nan_to_num(a.x_at_top, nan=-1),
                       np.nan_to_num(c.x_at_top, nan=-1))
    check("surrogate build is deterministic", same)

    P.clear_surrogate_cache()
    out = []
    errs = []
    def worker(hh):
        try:
            out.append(P.get_surrogate(hh))
        except Exception as e:      # noqa: BLE001
            errs.append(e)
    ths = [threading.Thread(target=worker, args=(P.in_to_m(10 + i % 4),)) for i in range(16)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    check("concurrent get_surrogate is safe", not errs and len(out) == 16, f"errors={errs}")
    check("cache respects its size cap", len(P._SUR_CACHE) <= P._SUR_MAX,
          f"size={len(P._SUR_CACHE)}")


# ── Timing baselines ────────────────────────────────────────────────────────
BASELINE = {"find_optimal": 1.621, "make_sweep": 1.055, "build_lut": 5.010, "simulate": 0.004748}


# ── 9. Shot family is exactly the set of shots that score ───────────────────
def test_family():
    bad_score = bad_window = bad_entry = 0
    total = 0
    worst_best = 0.0
    for h_in, d_in, drag, wind in CONFIGS[:4]:
        h, d = P.in_to_m(h_in), P.in_to_m(d_in)
        p = P.ShotParams(velocity=7.0, angle_deg=60.0, launch_height=h,
                         goal_distance=d, wind=wind, enable_drag=drag)
        fam = P.shot_family(p, dv_range=0.25, da_range=1.0, angle_step=1.0)
        if not fam["count"]:
            continue
        total += fam["count"]

        # Re-integrate the advertised commands and check they really score.
        lo_x, hi_x = P._entry_x_bounds(d, P.GOAL_DEPTH_M)
        r = P._integrate([s["velocity"] for s in fam["shots"]],
                         [s["angle_deg"] for s in fam["shots"]],
                         h, wind=wind, enable_drag=drag,
                         dt=P.DT_SURROGATE, stop_at_rim=True)
        bad_score += int((~P._made_from(r["x_at_top"], r["descending"], lo_x, hi_x)).sum())

        for s in fam["shots"]:
            if not (s["v_lo"] - 1e-9 <= s["velocity"] <= s["v_hi"] + 1e-9):
                bad_window += 1
            if s["entry_pos"] is None or not (0.0 <= s["entry_pos"] <= 1.0):
                bad_entry += 1

        # The family is quantised to whole degrees; find_optimal polishes in
        # 2-D, so it may be marginally better but must not be MUCH better.
        best = fam["shots"][fam["best_index"]]
        _, diag = P.find_optimal(d, h, enable_drag=drag, wind=wind,
                                 dv_range=0.25, da_range=1.0, with_diagnostics=True)
        worst_best = max(worst_best, diag["margin_sigma"] - best["margin_sigma"])

    check("every shot in the family scores", bad_score == 0, f"{bad_score}/{total} miss")
    check("family speed sits inside its own scoring window", bad_window == 0,
          f"{bad_window}/{total} outside")
    check("family entry position inside the opening", bad_entry == 0,
          f"{bad_entry}/{total} outside [0,1]")
    check("family best matches find_optimal", worst_best < 0.05,
          f"max Δmargin={worst_best:.4f}σ (1° quantisation)")


# ── 10. Monte-Carlo cloud agrees with the analytic probability ──────────────
def test_monte_carlo():
    worst_z = 0.0
    detail = ""
    for h_in, d_in, drag, wind in CONFIGS[:4]:
        h, d = P.in_to_m(h_in), P.in_to_m(d_in)
        best = P.find_optimal(d, h, enable_drag=drag, wind=wind,
                              dv_range=0.25, da_range=1.0)
        if best is None:
            continue
        p = P.ShotParams(velocity=best.velocity, angle_deg=best.angle_deg,
                         launch_height=h, goal_distance=d, wind=wind, enable_drag=drag)
        mc = P.monte_carlo(p, dv_range=0.25, da_range=1.0, n=600, seed=11)
        pa = mc["p_analytic"]
        se = math.sqrt(max(pa * (1.0 - pa), 1e-9) / mc["n"])
        z  = abs(mc["p_empirical"] - pa) / se
        if z > worst_z:
            worst_z, detail = z, f"{mc['p_empirical']:.3f} vs {pa:.3f}"
    check("Monte-Carlo cloud matches analytic P(make)", worst_z < 3.5,
          f"max z={worst_z:.2f} ({detail}, 600 samples)")


# ── 11. Batch distance sweep reproduces the single-distance solver ──────────
def test_target_sweep():
    h  = P.in_to_m(15.75)
    ds = [P.in_to_m(x) for x in range(24, 145, 8)]
    rows = P.sweep_targets(ds, h, dv_range=0.25, da_range=1.0)

    mism = 0
    for row in rows:
        best = P.find_optimal(row["distance_m"], h, dv_range=0.25, da_range=1.0)
        if best is None:
            mism += int(row["ok"])
            continue
        if (not row["ok"]
                or abs(row["velocity"]  - best.velocity)  > 1e-3
                or abs(row["angle_deg"] - best.angle_deg) > 1e-2):
            mism += 1
    check("sweep_targets reproduces find_optimal", mism == 0,
          f"{mism}/{len(rows)} rows differ")

    ok_rows = [r for r in rows if r["ok"]]

    # Speed alone is NOT monotone in distance: the optimiser is free to drop
    # the angle, which buys range at a lower speed.  The invariant is the
    # conditional one — hold the angle and the speed has to rise.
    drops = 0
    for a, b in zip(ok_rows, ok_rows[1:]):
        if abs(b["angle_deg"] - a["angle_deg"]) < 0.5 and b["velocity"] < a["velocity"] - 1e-6:
            drops += 1
    check("speed rises with distance when the angle is held", drops == 0,
          f"{drops} violations over {len(ok_rows)} reachable distances")

    # The scoring band narrows as the shot gets longer, so the best available
    # margin can only shrink.
    widened = sum(1 for a, b in zip(ok_rows, ok_rows[1:])
                  if b["margin_sigma"] > a["margin_sigma"] + 1e-6)
    check("robustness margin shrinks with distance", widened == 0,
          f"{widened} increases; {ok_rows[0]['margin_sigma']:.2f}σ → "
          f"{ok_rows[-1]['margin_sigma']:.2f}σ")


def bench():
    print("\n── timings (medians of 5; baseline = original implementation) ──")
    h, d = P.in_to_m(15.75), P.in_to_m(60)
    p = P.ShotParams(velocity=7.0, angle_deg=60, launch_height=h, goal_distance=d)

    def med(fn, n=5):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return sorted(ts)[len(ts) // 2]

    P.clear_surrogate_cache()
    t0 = time.perf_counter(); P.find_optimal(d, h); cold = time.perf_counter() - t0
    warm = med(lambda: P.find_optimal(d, h))
    sweep = med(lambda: P.make_sweep(p, dv_range=0.3, da_range=1.0, n=11))
    lut = med(lambda: P.build_lut(p))
    sim = med(lambda: P.simulate(p), 20)
    fam = med(lambda: P.shot_family(p, dv_range=0.3, da_range=1.0, angle_step=1.0))
    mc  = med(lambda: P.monte_carlo(p, dv_range=0.3, da_range=1.0, n=250, seed=2))
    tsw = med(lambda: P.sweep_targets([P.in_to_m(x) for x in range(24, 145, 8)], h,
                                      dv_range=0.3, da_range=1.0))

    rows = [
        ("find_optimal (cold cache)", cold,  BASELINE["find_optimal"]),
        ("find_optimal (warm cache)", warm,  BASELINE["find_optimal"]),
        ("make_sweep",                sweep, BASELINE["make_sweep"]),
        ("build_lut",                 lut,   BASELINE["build_lut"]),
        ("simulate",                  sim,   BASELINE["simulate"]),
    ]
    for name, got, base in rows:
        print(f"  {name:28s} {got*1000:9.2f} ms   baseline {base*1000:9.2f} ms   "
              f"{base/got:7.1f}x")

    # No baseline for these — they are new, not rewritten.
    print("\n── timings (new views; no prior implementation to compare against) ──")
    for name, got in (("shot_family (1°)", fam),
                      ("monte_carlo (250)", mc),
                      ("sweep_targets (16 d)", tsw)):
        print(f"  {name:28s} {got*1000:9.2f} ms")


if __name__ == "__main__":
    print("── invariants ──")
    test_cd()
    test_event_convergence()
    test_monotone()
    test_band()
    test_p_make()
    test_cross_path()
    test_optimum()
    test_cache()
    test_family()
    test_monte_carlo()
    test_target_sweep()
    bench()
    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
    raise SystemExit(1 if FAILS else 0)
