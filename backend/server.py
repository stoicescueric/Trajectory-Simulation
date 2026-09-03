"""
Flask backend for DECODE Shot Simulator.
Endpoints:
  POST /api/simulate  -> single trajectory
  POST /api/sweep     -> spaghetti sweep (all trajectories)
  POST /api/family    -> every scoring shot at this distance, one per angle
  POST /api/montecarlo-> N uniform samples from a bounded perturbation box
  POST /api/target_sweep -> best shot at each of a chunk of distances
  POST /api/optimize  -> find optimal (v, angle)
  POST /api/lut       -> download CSV lookup table
"""
import csv
import io
import math
import os

from flask import Flask, request, jsonify, send_from_directory, Response

from physics import (
    ShotParams, simulate, make_sweep, find_optimal, build_lut,
    shot_family, monte_carlo, sweep_targets,
    in_to_m, GOAL_TOP_HEIGHT_M, GOAL_DEPTH_M,
)

app = Flask(__name__, static_folder=None)


@app.errorhandler(ValueError)
def invalid_request(exc):
    return jsonify({"ok": False, "error": str(exc)}), 400

try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    pass

FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)


def _params(d: dict) -> ShotParams:
    p = ShotParams(
        velocity      = float(d.get("velocity",      5.0)),
        angle_deg     = float(d.get("angle_deg",     60.0)),
        spin_rpm      = 0.0,
        launch_height = float(d.get("launch_height", in_to_m(15.75))),
        launch_x      = float(d.get("launch_x",      0.0)),
        goal_distance = float(d.get("goal_distance", in_to_m(60.0))),
        goal_height   = float(d.get("goal_height",   GOAL_TOP_HEIGHT_M)),
        goal_depth    = float(d.get("goal_depth",    GOAL_DEPTH_M)),
        wind          = float(d.get("wind",          0.0)),
        enable_drag   = bool(d.get("enable_drag",    True)),
        enable_magnus = False,
    )
    numeric = (
        p.velocity, p.angle_deg, p.launch_height, p.launch_x,
        p.goal_distance, p.goal_height, p.goal_depth, p.wind,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("trajectory inputs must be finite numbers")
    if p.velocity <= 0 or p.launch_height < 0 or p.goal_distance <= 0:
        raise ValueError("velocity and distance must be positive; launch height cannot be negative")
    return p


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    p = _params(request.json or {})
    r = simulate(p)
    step = max(1, len(r.t) // 200)
    return jsonify({
        "t":  r.t[::step],  "x":  r.x[::step],  "y":  r.y[::step],
        "vx": r.vx[::step], "vy": r.vy[::step],
        "apex_x": r.apex_x, "apex_y": r.apex_y,
        "range_x": r.range_x,
        "impact_y_at_goal": r.impact_y_at_goal,
        "x_at_top":         r.x_at_top,
        "t_at_top":         r.t_at_top,
        "entry_angle_deg":  r.entry_angle_deg,
        "made": r.made,
        "goal": {"distance": p.goal_distance, "height": p.goal_height, "depth": p.goal_depth},
        "launch_height": p.launch_height,
    })


@app.route("/api/sweep", methods=["POST"])
def api_sweep():
    body = request.json or {}
    p  = _params(body)
    dv = float(body.get("dv_range",   0.7))
    da = float(body.get("da_range",   1.5))
    n  = int(body.get("resolution",  11))
    n  = max(3, min(21, n))
    return jsonify(make_sweep(p, dv_range=dv, da_range=da, n=n))


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    body     = request.json or {}
    p        = _params(body)
    distance = p.goal_distance
    h_launch = p.launch_height
    drag     = p.enable_drag
    # dv_range / da_range are bounded engineering half-widths.
    dv_range = float(body.get("dv_range",      0.7))
    da_range = float(body.get("da_range",      1.5))
    best, diag = find_optimal(distance, h_launch, enable_drag=drag,
                              dv_range=dv_range, da_range=da_range,
                              goal_height=p.goal_height, goal_depth=p.goal_depth,
                              wind=p.wind, with_diagnostics=True)
    if best is None:
        return jsonify({"ok": False})
    return jsonify({
        "ok":           True,
        "velocity":     best.velocity,
        "angle_deg":    best.angle_deg,
        "window_pct":   diag["window_pct"],
        "coverage":     diag["coverage"],
        "margin_sigma": diag["margin_sigma"],
        "p_make":       diag["p_make"],
        "margins": {
            "v_down_ms":  diag["v_down_ms"],
            "v_up_ms":    diag["v_up_ms"],
            "a_down_deg": diag["a_down_deg"],
            "a_up_deg":   diag["a_up_deg"],
        },
        "monotone_ok":  diag["monotone_ok"],
    })


@app.route("/api/family", methods=["POST"])
def api_family():
    body = request.json or {}
    p = _params(body)
    return jsonify(shot_family(
        p,
        dv_range   = float(body.get("dv_range",   0.7)),
        da_range   = float(body.get("da_range",   1.5)),
        angle_step = float(body.get("angle_step", 1.0)),
    ))


@app.route("/api/montecarlo", methods=["POST"])
def api_montecarlo():
    body = request.json or {}
    p = _params(body)
    seed = body.get("seed")
    return jsonify(monte_carlo(
        p,
        dv_range = float(body.get("dv_range", 0.7)),
        da_range = float(body.get("da_range", 1.5)),
        n        = int(body.get("samples",  250)),
        seed     = None if seed is None else int(seed),
    ))


# Deliberately stateless: the client sends one chunk of distances per request
# and owns the progress bar.  No job registry, no streaming, and — because the
# surrogate is cached across requests — no cost to splitting the work up.
MAX_SWEEP_CHUNK = 32


@app.route("/api/target_sweep", methods=["POST"])
def api_target_sweep():
    try:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")

        raw = body.get("distances")
        if not isinstance(raw, list) or not raw:
            raise ValueError("distances must be a non-empty list")
        if len(raw) > MAX_SWEEP_CHUNK:
            raise ValueError(f"at most {MAX_SWEEP_CHUNK} distances per request")

        distances = []
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("each distance must be a finite number")
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("each distance must be a positive, finite number")
            distances.append(value)

        p = _params(body)
        rows = sweep_targets(
            distances, p.launch_height,
            enable_drag = p.enable_drag,
            dv_range    = float(body.get("dv_range", 0.7)),
            da_range    = float(body.get("da_range", 1.5)),
            goal_height = p.goal_height,
            goal_depth  = p.goal_depth,
            wind        = p.wind,
        )
        return jsonify({"ok": True, "rows": rows})
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/lut", methods=["POST"])
def api_lut():
    body = request.json or {}
    p = _params(body)
    rows = build_lut(p)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

    filename = (
        f"decode_lut_d{p.goal_distance*100:.0f}cm"
        f"_h{p.launch_height*100:.0f}cm"
        f"_drag{'on' if p.enable_drag else 'off'}.csv"
    )
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
