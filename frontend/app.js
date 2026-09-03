/* DECODE Shot Simulator — App Logic */

const API = async (path, body) => {
  const response = await fetch(`/api/${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `request failed (${response.status})`);
  return data;
};

const IN2M = 0.0254;
const MS2MPH = 2.23694;
const MS2FPS = 3.28084;

const $ = id => document.getElementById(id);

const ui = {
  velocity: $('velocity'), angle: $('angle'),
  height: $('height'), distance: $('distance'), wind: $('wind'),
  drag: $('drag'),
  vd: $('v-display'), ad: $('a-display'),
  hd: $('h-display'), dd: $('d-display'), wd: $('w-display'),
  vmph: $('v-mph'), vfps: $('v-fps'),
  dtiles: $('d-tiles'), dm: $('d-m'),
  dvd: $('dv-display'), dad: $('da-display'),
  sweepDv: $('sweep-dv'), sweepDa: $('sweep-da'),
  resolution: $('resolution'),
  trajCanvas: $('trajectory'),
  isoCanvas: $('traj3d'),
  sweepCanvas: $('sweep-canvas'),
  tolmapCanvas: $('tolmap-canvas'),
  rdApex: $('rd-apex'), rdTime: $('rd-time'),
  rdImpact: $('rd-impact'), rdEntry: $('rd-entry'),
  rdMade: $('rd-made'), rdWindow: $('rd-window'),
  rdMargin: $('rd-margin'), rdSlack: $('rd-slack'),
  resultReadout: $('result-readout'),
  btnOpt: $('btn-optimize'), btnWin: $('btn-window'),
  btnDownload: $('btn-download'),
  optOut: $('optimize-out'),
  familyCanvas: $('family-canvas'), targetsCanvas: $('targets-canvas'),
  familyTip: $('family-tip'), familyList: $('family-list'),
  familyStatus: $('family-status'), familyStep: $('family-step'),
  familyModeCtl: $('family-mode'), mcSamples: $('mc-samples'),
  btnFamily: $('btn-family'),
  tsFrom: $('ts-from'), tsTo: $('ts-to'), tsStep: $('ts-step'),
  tsFill: $('ts-fill'), tsStatus: $('ts-status'),
  btnTargets: $('btn-targets'), btnTargetsCsv: $('btn-targets-csv'),
  az: $('az'), el: $('el'),
  tabs: document.querySelectorAll('.tab'),
  tabPanels: {
    side: $('tab-side'), iso: $('tab-iso'),
    sweep: $('tab-sweep'), tolmap: $('tab-tolmap'),
    family: $('tab-family'), targets: $('tab-targets'),
  },
};

// ── Param helpers ──────────────────────────────────────────────────────────

function readParams() {
  return {
    velocity:      +ui.velocity.value,
    angle_deg:     +ui.angle.value,
    spin_rpm:      0,
    launch_height: +ui.height.value * IN2M,
    goal_distance: +ui.distance.value * IN2M,
    wind:          +ui.wind.value,
    enable_drag:   ui.drag.checked,
    enable_magnus: false,
  };
}

function setSliderFill(el) {
  const pct = ((+el.value - +el.min) / (+el.max - +el.min)) * 100;
  el.style.setProperty('--pct', pct + '%');
}

function updateLabels() {
  const v = +ui.velocity.value;
  ui.vd.innerHTML = `${v.toFixed(2)} <em>m/s</em>`;
  ui.vmph.textContent = `${(v * MS2MPH).toFixed(2)} mph`;
  ui.vfps.textContent = `${(v * MS2FPS).toFixed(2)} ft/s`;
  setSliderFill(ui.velocity);

  const a = +ui.angle.value;
  ui.ad.innerHTML = `${a.toFixed(1)}<em>°</em>`;
  setSliderFill(ui.angle);

  const h = +ui.height.value;
  ui.hd.innerHTML = `${h.toFixed(2)} <em>in</em>`;
  setSliderFill(ui.height);

  const d = +ui.distance.value;
  ui.dd.innerHTML = `${d.toFixed(1)} <em>in</em>`;
  ui.dtiles.textContent = `${(d / 24).toFixed(2)} tiles`;
  ui.dm.textContent = `${(d * IN2M).toFixed(3)} m`;
  setSliderFill(ui.distance);

  const w = +ui.wind.value;
  ui.wd.innerHTML = `${w >= 0 ? '+' : ''}${w.toFixed(1)} <em>m/s</em>`;
  setSliderFill(ui.wind);

  const dv = +ui.sweepDv.value;
  ui.dvd.innerHTML = `${dv.toFixed(2)} <em>m/s</em>`;
  setSliderFill(ui.sweepDv);

  const da = +ui.sweepDa.value;
  ui.dad.innerHTML = `${da.toFixed(1)} <em>°</em>`;
  setSliderFill(ui.sweepDa);
}

// Statistics arrive from the server; the UI only formats them.
function showSweepStats(d) {
  if (!d) return;
  ui.rdWindow.textContent = ((d.coverage ?? d.p_make) * 100).toFixed(1) + '%';
  ui.rdMargin.textContent = d.margin_sigma >= 0
    ? `${d.margin_sigma.toFixed(2)}× half-width`
    : 'MISSES';
  const m = d.margins || {};
  const f = (z, u, dp) => (z == null ? '—' : `${z.toFixed(dp)}${u}`);
  ui.rdSlack.textContent =
    `−${f(m.v_down_ms, '', 2)}/+${f(m.v_up_ms, ' m/s', 2)} · ` +
    `−${f(m.a_down_deg, '', 1)}/+${f(m.a_up_deg, '°', 1)}`;
}

// ── Tab switching ──────────────────────────────────────────────────────────

let activeTab = 'side';

function redrawTab(name) {
  if (name === 'side'    && lastTraj)  drawTrajectory(lastTraj);
  if (name === 'iso'     && lastTraj)  draw3D(lastTraj);
  if (name === 'sweep'   && lastSweep) drawSweep(lastSweep);
  if (name === 'tolmap'  && lastSweep) drawToleranceMap(lastSweep);
  if (name === 'family')               drawFamilyTab();
  if (name === 'targets')              drawTargets(lastTargets);
}

function showTab(name) {
  activeTab = name;
  ui.tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  Object.entries(ui.tabPanels).forEach(([k, el]) => el.classList.toggle('hidden', k !== name));
  redrawTab(name);
}

ui.tabs.forEach(btn => {
  btn.addEventListener('click', () => showTab(btn.dataset.tab));
});

// ── Canvas resize helper ────────────────────────────────────────────────────

function fitCanvas(canvas) {
  const dpr  = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.floor(rect.width  * dpr);
  const h = Math.floor(rect.height * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width  = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W: rect.width, H: rect.height };
}

// ══════════════════════════════════════════════════════════════════════════
// 2D TRAJECTORY CANVAS
// ══════════════════════════════════════════════════════════════════════════

let lastTraj = null;

function drawTrajectory(data) {
  const { ctx, W, H } = fitCanvas(ui.trajCanvas);
  ctx.clearRect(0, 0, W, H);

  const padL = 54, padR = 30, padT = 28, padB = 38;
  const xMax = Math.max(data.goal.distance + data.goal.depth + 0.4, ...data.x) + 0.15;
  const yMax = Math.max(1.6, data.apex_y + 0.3, data.goal.height + 0.4);

  const wPx = W - padL - padR;
  const hPx = H - padT - padB;
  const X = x => padL + x / xMax * wPx;
  const Y = y => H - padB - y / yMax * hPx;

  // grid
  const drawGrid = (step, alpha) => {
    ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
    ctx.lineWidth = 1;
    for (let x = 0; x <= xMax; x += step) {
      ctx.beginPath(); ctx.moveTo(X(x), padT); ctx.lineTo(X(x), H - padB); ctx.stroke();
    }
    for (let y = 0; y <= yMax; y += step) {
      ctx.beginPath(); ctx.moveTo(padL, Y(y)); ctx.lineTo(W - padR, Y(y)); ctx.stroke();
    }
  };
  drawGrid(0.25, 0.025);
  drawGrid(1.0,  0.055);

  // axes
  ctx.strokeStyle = '#2a3a4e';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB); ctx.lineTo(W - padR, H - padB);
  ctx.stroke();

  // tick labels
  ctx.fillStyle = '#637587';
  ctx.font = '10px JetBrains Mono';
  ctx.textAlign = 'center';
  for (let x = 0; x <= xMax; x += 0.5) {
    ctx.fillText(x.toFixed(1) + 'm', X(x), H - padB + 14);
  }
  ctx.textAlign = 'right';
  for (let y = 0; y <= yMax; y += 0.25) {
    ctx.fillText(y.toFixed(2), padL - 5, Y(y) + 3);
  }

  // axis labels
  ctx.fillStyle = '#637587';
  ctx.font = '9px JetBrains Mono';
  ctx.textAlign = 'right';
  ctx.fillText('DISTANCE (m) →', W - padR, H - padB + 28);
  ctx.save();
  ctx.translate(13, padT + hPx / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('HEIGHT (m)', 0, 0);
  ctx.restore();

  // floor shading
  ctx.fillStyle = 'rgba(43,128,255,0.03)';
  ctx.fillRect(padL, Y(0), wPx, H - padB - Y(0));

  // goal basket — drawn as a U-shaped box open at the top
  const gx  = X(data.goal.distance);
  const gy  = Y(data.goal.height);
  const gd  = data.goal.depth * (wPx / xMax);
  const gx2 = gx + gd;
  // interior fill
  ctx.fillStyle = 'rgba(0,229,160,0.07)';
  ctx.fillRect(gx, gy, gd, H - padB - gy);
  // walls: front face, back face, floor of basket
  ctx.strokeStyle = 'rgba(0,229,160,0.7)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(gx,  H - padB);   // front wall bottom
  ctx.lineTo(gx,  gy);          // front wall up to rim
  ctx.lineTo(gx2, gy);          // top opening (rim)
  ctx.lineTo(gx2, H - padB);   // back wall down
  ctx.stroke();
  // label
  ctx.fillStyle = '#00e5a0';
  ctx.font = '9px JetBrains Mono';
  ctx.textAlign = 'left';
  ctx.fillText(`GOAL  top: ${data.goal.height.toFixed(2)} m`, gx + 4, gy - 6);

  // robot silhouette
  const lx = X(0);
  const ly = Y(data.launch_height);
  ctx.fillStyle = 'rgba(19,25,32,0.9)';
  ctx.fillRect(lx - 15, ly, 30, H - padB - ly);
  ctx.strokeStyle = '#2a3a4e';
  ctx.lineWidth = 1;
  ctx.strokeRect(lx - 15, ly, 30, H - padB - ly);
  ctx.fillStyle = '#00d4ff';
  ctx.beginPath(); ctx.arc(lx, ly, 3.5, 0, Math.PI * 2); ctx.fill();

  // trajectory glow + line
  ctx.strokeStyle = 'rgba(0,212,255,0.12)';
  ctx.lineWidth = 9;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  data.x.forEach((xi, i) => {
    i === 0 ? ctx.moveTo(X(xi), Y(data.y[i])) : ctx.lineTo(X(xi), Y(data.y[i]));
  });
  ctx.stroke();

  ctx.strokeStyle = '#00d4ff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  data.x.forEach((xi, i) => {
    i === 0 ? ctx.moveTo(X(xi), Y(data.y[i])) : ctx.lineTo(X(xi), Y(data.y[i]));
  });
  ctx.stroke();

  // time dots every ~0.1 s
  if (data.t.length > 1) {
    const dt = data.t[1] - data.t[0];
    const stride = Math.max(1, Math.round(0.1 / dt));
    ctx.fillStyle = '#00d4ff';
    for (let i = 0; i < data.x.length; i += stride) {
      ctx.beginPath(); ctx.arc(X(data.x[i]), Y(data.y[i]), 1.5, 0, Math.PI * 2); ctx.fill();
    }
  }

  // apex marker
  ctx.fillStyle = '#ffb547';
  ctx.beginPath(); ctx.arc(X(data.apex_x), Y(data.apex_y), 5, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = 'rgba(255,181,71,0.4)';
  ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
  ctx.beginPath(); ctx.moveTo(X(data.apex_x), Y(data.apex_y)); ctx.lineTo(X(data.apex_x), H - padB); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#ffb547';
  ctx.font = '9px JetBrains Mono';
  ctx.textAlign = 'left';
  ctx.fillText(`APEX ${data.apex_y.toFixed(2)} m`, X(data.apex_x) + 7, Y(data.apex_y) - 5);

  // impact dot — placed where ball descends through top plane (x_at_top, goal_height)
  // Falls back to front-face crossing if top crossing wasn't recorded yet.
  if (data.x_at_top != null) {
    const dotX = X(data.x_at_top);
    const dotY = gy;   // y = goal_height (the top rim line)
    ctx.fillStyle = data.made ? '#00e5a0' : '#ff4d6a';
    ctx.shadowColor = data.made ? '#00e5a0' : '#ff4d6a';
    ctx.shadowBlur = 10;
    ctx.beginPath(); ctx.arc(dotX, dotY, 6, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
  } else if (data.impact_y_at_goal != null) {
    ctx.fillStyle = '#ff4d6a';
    ctx.shadowColor = '#ff4d6a'; ctx.shadowBlur = 8;
    ctx.beginPath(); ctx.arc(gx, Y(data.impact_y_at_goal), 5, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
  }

}

function showTrajStats(data) {
  ui.rdApex.textContent   = `${data.apex_y.toFixed(2)} m`;
  ui.rdTime.textContent   = data.t_at_top == null ? '—' : `${data.t_at_top.toFixed(3)} s`;
  // "Y @ GOAL" shows height at the front face; entry shows angle at top crossing
  ui.rdImpact.textContent = data.impact_y_at_goal != null
    ? `${data.impact_y_at_goal.toFixed(3)} m`
    : (data.x_at_top != null ? `over rim` : 'N/A');
  ui.rdEntry.textContent  = data.entry_angle_deg  != null ? `${data.entry_angle_deg.toFixed(1)} °` : '—';

  ui.resultReadout.classList.remove('made', 'miss');
  if (data.x_at_top == null && data.impact_y_at_goal == null) {
    ui.rdMade.textContent = 'NO REACH';
    ui.resultReadout.classList.add('miss');
  } else if (data.made) {
    ui.rdMade.textContent = '✓ SCORE';
    ui.resultReadout.classList.add('made');
  } else {
    ui.rdMade.textContent = '✗ MISS';
    ui.resultReadout.classList.add('miss');
  }
}

// ══════════════════════════════════════════════════════════════════════════
// SHARED SIDE-PLANE FRAME
// ══════════════════════════════════════════════════════════════════════════
// Grid, axes, floor and goal basket, in world metres.  Every view that draws
// trajectories in the side plane goes through this, so they stay consistent
// with each other and there is one place to change the field drawing.

function worldExtents(paths, goal) {
  let xMax = 0, yMax = 0;
  for (const p of paths) {
    for (const xi of p.x) if (xi > xMax) xMax = xi;
    for (const yi of p.y) if (yi > yMax) yMax = yi;
  }
  return {
    xMax: Math.max(xMax, goal.distance + goal.depth + 0.3) + 0.1,
    yMax: Math.max(yMax, goal.height + 0.5) + 0.1,
  };
}

function drawFieldFrame(canvas, { xMax, yMax, goal }) {
  const { ctx, W, H } = fitCanvas(canvas);
  ctx.clearRect(0, 0, W, H);

  const padL = 54, padR = 30, padT = 28, padB = 38;
  const wPx = W - padL - padR;
  const hPx = H - padT - padB;
  const X = x => padL + x / xMax * wPx;
  const Y = y => H - padB - y / yMax * hPx;

  // grid
  const drawGrid = (step, alpha) => {
    ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
    ctx.lineWidth = 1;
    for (let x = 0; x <= xMax; x += step) {
      ctx.beginPath(); ctx.moveTo(X(x), padT); ctx.lineTo(X(x), H - padB); ctx.stroke();
    }
    for (let y = 0; y <= yMax; y += step) {
      ctx.beginPath(); ctx.moveTo(padL, Y(y)); ctx.lineTo(W - padR, Y(y)); ctx.stroke();
    }
  };
  drawGrid(0.25, 0.022);
  drawGrid(1.0,  0.05);

  // axes
  ctx.strokeStyle = '#2a3a4e'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB); ctx.lineTo(W - padR, H - padB);
  ctx.stroke();

  // tick labels
  ctx.fillStyle = '#637587'; ctx.font = '10px JetBrains Mono';
  ctx.textAlign = 'center';
  for (let x = 0; x <= xMax; x += 0.5) ctx.fillText(x.toFixed(1) + 'm', X(x), H - padB + 14);
  ctx.textAlign = 'right';
  for (let y = 0; y <= yMax; y += 0.25) ctx.fillText(y.toFixed(2), padL - 5, Y(y) + 3);

  // floor
  ctx.fillStyle = 'rgba(43,128,255,0.025)';
  ctx.fillRect(padL, Y(0), wPx, H - padB - Y(0));

  // goal basket — U-shaped box
  const gx  = X(goal.distance);
  const gy  = Y(goal.height);
  const gd  = goal.depth * (wPx / xMax);
  const gx2 = gx + gd;
  ctx.fillStyle = 'rgba(0,229,160,0.06)';
  ctx.fillRect(gx, gy, gd, H - padB - gy);
  ctx.strokeStyle = 'rgba(0,229,160,0.7)'; ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(gx,  H - padB);
  ctx.lineTo(gx,  gy);
  ctx.lineTo(gx2, gy);
  ctx.lineTo(gx2, H - padB);
  ctx.stroke();

  return { ctx, W, H, X, Y, padL, padR, padT, padB, wPx, hPx, gx, gy, gx2 };
}

// Trace one path in world coordinates.  Caller sets the stroke style.
function strokePath(ctx, X, Y, p) {
  ctx.beginPath();
  for (let i = 0; i < p.x.length; i++) {
    const px = X(p.x[i]), py = Y(p.y[i]);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.stroke();
}

function emptyCanvasMessage(canvas, message) {
  const { ctx, W, H } = fitCanvas(canvas);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#637587';
  ctx.font = '12px JetBrains Mono';
  ctx.textAlign = 'center';
  ctx.fillText(message, W / 2, H / 2);
  ctx.textAlign = 'left';
}

// ══════════════════════════════════════════════════════════════════════════
// SPAGHETTI SWEEP CANVAS
// ══════════════════════════════════════════════════════════════════════════

let lastSweep = null;

function drawSweep(sweepData) {
  const trajs = sweepData.trajectories;
  if (!trajs || !trajs.length) {
    const blank = fitCanvas(ui.sweepCanvas);
    blank.ctx.clearRect(0, 0, blank.W, blank.H);
    return;
  }

  const { xMax, yMax } = worldExtents(trajs, sweepData.goal);
  const { ctx, W, H, X, Y, padL, padR, padT, padB, gx } =
    drawFieldFrame(ui.sweepCanvas, { xMax, yMax, goal: sweepData.goal });

  // draw miss trajectories first (faint red), then makes (bright green)
  const drawGroup = (arr, color, alpha, width) => {
    ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = width;
    for (const tr of arr) {
      ctx.beginPath();
      tr.x.forEach((xi, i) => {
        i === 0 ? ctx.moveTo(X(xi), Y(tr.y[i])) : ctx.lineTo(X(xi), Y(tr.y[i]));
      });
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  };

  const rings  = trajs.filter(t => t.kind !== 'boundary');
  const bounds = trajs.filter(t => t.kind === 'boundary');
  const misses = rings.filter(t => !t.made);
  const makes  = rings.filter(t =>  t.made);

  drawGroup(misses, '#ff4d6a', 0.20, 1.0);
  drawGroup(makes,  '#00e5a0', 0.65, 1.5);

  // Shots sitting exactly on the make/miss edge — the most informative curves
  // in the plot, so they get their own dashed styling.
  ctx.setLineDash([4, 3]);
  drawGroup(bounds, '#ffc857', 0.85, 1.4);
  ctx.setLineDash([]);

  // impact dots
  for (const tr of trajs) {
    if (tr.impact_y_at_goal == null) continue;
    ctx.fillStyle = tr.made ? 'rgba(0,229,160,0.8)' : 'rgba(255,77,106,0.5)';
    ctx.beginPath(); ctx.arc(gx, Y(tr.impact_y_at_goal), tr.made ? 3 : 2, 0, Math.PI * 2); ctx.fill();
  }

  // nominal trajectory on top
  if (sweepData.nominal) {
    const n = sweepData.nominal;
    ctx.strokeStyle = '#00d4ff'; ctx.lineWidth = 2.5; ctx.globalAlpha = 1;
    ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 8;
    ctx.beginPath();
    n.x.forEach((xi, i) => {
      i === 0 ? ctx.moveTo(X(xi), Y(n.y[i])) : ctx.lineTo(X(xi), Y(n.y[i]));
    });
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // legend
  ctx.font = '10px JetBrains Mono'; ctx.globalAlpha = 1;
  const lx = padL + 10, ly = padT + 12;
  ctx.fillStyle = '#00e5a0'; ctx.fillRect(lx, ly, 14, 2);
  ctx.fillStyle = '#b4c2d0'; ctx.textAlign = 'left';
  ctx.fillText(`SCORE  (${makes.length})`, lx + 18, ly + 4);
  ctx.fillStyle = '#ff4d6a'; ctx.fillRect(lx, ly + 14, 14, 2);
  ctx.fillStyle = '#b4c2d0';
  ctx.fillText(`MISS   (${misses.length})`, lx + 18, ly + 18);
  ctx.fillStyle = '#ffc857'; ctx.fillRect(lx, ly + 28, 14, 2);
  ctx.fillStyle = '#b4c2d0';
  ctx.fillText(`EDGE   (${bounds.length})`, lx + 18, ly + 32);
  ctx.fillStyle = '#00d4ff'; ctx.fillRect(lx, ly + 42, 14, 2);
  ctx.fillStyle = '#b4c2d0';
  ctx.fillText('NOMINAL', lx + 18, ly + 46);

  showSweepStats(sweepData);

  // axes labels
  ctx.fillStyle = '#637587'; ctx.font = '9px JetBrains Mono';
  ctx.textAlign = 'right';
  ctx.fillText('DISTANCE (m) →', W - padR, H - padB + 28);
}

// ══════════════════════════════════════════════════════════════════════════
// TOLERANCE MAP  —  the scoring region in (∆angle, ∆velocity) space
// ══════════════════════════════════════════════════════════════════════════
// The server sends the make-region as two boundary curves rather than a pixel
// mask, so this renders as a crisp filled band and stays tiny over the wire.

function drawToleranceMap(d) {
  const { ctx, W, H } = fitCanvas(ui.tolmapCanvas);
  ctx.clearRect(0, 0, W, H);
  if (!d || !d.band || !(d.half_width || d.sigma) || !d.nominal_shot) return;

  const halfWidth = d.half_width || d.sigma;
  const sv = halfWidth.v, sa = halfWidth.a;
  const v0 = d.nominal_shot.velocity, a0 = d.nominal_shot.angle_deg;

  const padL = 66, padR = 28, padT = 26, padB = 46;
  const wPx = W - padL - padR, hPx = H - padT - padB;
  if (wPx <= 10 || hPx <= 10) return;

  const NS = 3.5;                       // view radius in declared half-widths
  const AR = NS * sa, VR = NS * sv;
  const X = da => padL + (da + AR) / (2 * AR) * wPx;
  const Y = dv => padT + (VR - dv) / (2 * VR) * hPx;

  // ── grid at whole-σ intervals ───────────────────────────────────────────
  ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 1;
  for (let k = -3; k <= 3; k++) {
    ctx.beginPath(); ctx.moveTo(X(k * sa), padT); ctx.lineTo(X(k * sa), padT + hPx); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(padL, Y(k * sv)); ctx.lineTo(padL + wPx, Y(k * sv)); ctx.stroke();
  }

  // ── the make region ─────────────────────────────────────────────────────
  ctx.save();
  ctx.beginPath(); ctx.rect(padL, padT, wPx, hPx); ctx.clip();

  const runs = [];
  let cur = null;
  for (let i = 0; i < d.band.angle_deg.length; i++) {
    const lo = d.band.v_lo[i], hi = d.band.v_hi[i];
    const da = d.band.angle_deg[i] - a0;
    if (lo == null || hi == null || Math.abs(da) > AR * 1.08) { cur = null; continue; }
    if (!cur) { cur = []; runs.push(cur); }
    cur.push([da, lo - v0, hi - v0]);
  }

  for (const run of runs) {
    if (run.length < 2) continue;
    ctx.beginPath();
    run.forEach(([da, , hi], i) => (i ? ctx.lineTo(X(da), Y(hi)) : ctx.moveTo(X(da), Y(hi))));
    for (let i = run.length - 1; i >= 0; i--) ctx.lineTo(X(run[i][0]), Y(run[i][1]));
    ctx.closePath();
    ctx.fillStyle = 'rgba(0,229,160,0.16)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,229,160,0.85)'; ctx.lineWidth = 1.6;
    ctx.stroke();
  }

  // ── σ ellipses: where the shooter's error actually lands ────────────────
  for (const k of [1, 2, 3]) {
    ctx.strokeStyle = `rgba(0,212,255,${k === 1 ? 0.75 : k === 2 ? 0.4 : 0.22})`;
    ctx.lineWidth = k === 1 ? 1.6 : 1;
    ctx.setLineDash(k === 1 ? [] : [3, 3]);
    ctx.strokeRect(X(-k * sa), Y(k * sv),
                   X(k * sa) - X(-k * sa), Y(-k * sv) - Y(k * sv));
    ctx.setLineDash([]);
  }

  // ── nominal shot ────────────────────────────────────────────────────────
  ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(0), padT); ctx.lineTo(X(0), padT + hPx); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(padL + wPx, Y(0)); ctx.stroke();
  ctx.fillStyle = '#00d4ff';
  ctx.beginPath(); ctx.arc(X(0), Y(0), 4, 0, Math.PI * 2); ctx.fill();
  ctx.restore();

  // ── frame + ticks ───────────────────────────────────────────────────────
  ctx.strokeStyle = '#2a3a4e'; ctx.lineWidth = 1.5;
  ctx.strokeRect(padL, padT, wPx, hPx);

  ctx.fillStyle = '#637587'; ctx.font = '10px JetBrains Mono';
  ctx.textAlign = 'center';
  for (let k = -3; k <= 3; k++) {
    ctx.fillText(`${k > 0 ? '+' : ''}${(k * sa).toFixed(1)}°`, X(k * sa), padT + hPx + 15);
    if (k !== 0) ctx.fillText(`${k}×`, X(k * sa), padT + hPx + 27);
  }
  ctx.textAlign = 'right';
  for (let k = -3; k <= 3; k++) {
    ctx.fillText(`${k > 0 ? '+' : ''}${(k * sv).toFixed(2)}`, padL - 8, Y(k * sv) + 3);
  }

  ctx.fillStyle = '#637587'; ctx.font = '9px JetBrains Mono';
  ctx.textAlign = 'center';
  ctx.fillText('∆ LAUNCH ANGLE →', padL + wPx / 2, H - 6);
  ctx.save();
  ctx.translate(14, padT + hPx / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('∆ EXIT VELOCITY (m/s) →', 0, 0);
  ctx.restore();

  // ── legend / headline number ────────────────────────────────────────────
  ctx.font = '10px JetBrains Mono'; ctx.textAlign = 'left';
  const lx = padL + 12, ly = padT + 14;
  ctx.fillStyle = 'rgba(0,229,160,0.85)'; ctx.fillRect(lx, ly - 4, 14, 8);
  ctx.fillStyle = '#b4c2d0'; ctx.fillText('SCORES', lx + 20, ly + 3);
  ctx.strokeStyle = 'rgba(0,212,255,0.75)'; ctx.lineWidth = 1.6;
  ctx.beginPath(); ctx.moveTo(lx, ly + 14); ctx.lineTo(lx + 14, ly + 14); ctx.stroke();
  ctx.fillStyle = '#b4c2d0'; ctx.fillText('1× / 2× / 3× PERTURBATION BOX', lx + 20, ly + 17);

  ctx.textAlign = 'right';
  ctx.font = '600 13px JetBrains Mono';
  const coverage = d.coverage ?? d.p_make;
  ctx.fillStyle = coverage > 0.8 ? '#00e5a0' : coverage > 0.5 ? '#ffc857' : '#ff4d6a';
  ctx.fillText(`ROBUST COVERAGE ${(coverage * 100).toFixed(1)}%`, padL + wPx - 12, padT + 18);

  showSweepStats(d);
}

// ══════════════════════════════════════════════════════════════════════════
// 3D ISOMETRIC CANVAS  (pure canvas projection, no WebGL)
// ══════════════════════════════════════════════════════════════════════════

function draw3D(data) {
  const { ctx, W, H } = fitCanvas(ui.isoCanvas);
  ctx.clearRect(0, 0, W, H);

  const az = (+ui.az.value) * Math.PI / 180;
  const el = (+ui.el.value) * Math.PI / 180;

  // project 3D (x,y,z) => 2D screen. x=forward, y=height, z=lateral (0 for planar)
  const scale = Math.min(W, H) * 0.7;

  function project(px, py, pz) {
    // rotation around vertical axis (az) then tilt (el)
    const cosA = Math.cos(az), sinA = Math.sin(az);
    const rx = px * cosA + pz * sinA;
    const rz = -px * sinA + pz * cosA;
    const cosE = Math.cos(el), sinE = Math.sin(el);
    const sx = rx;
    const sy = py * cosE - rz * sinE;
    const sz = py * sinE + rz * cosE;
    const fov = 2.5;
    const s = scale / (sz + fov);
    return { sx: W / 2 + sx * s, sy: H / 2 - sy * s };
  }

  // normalize trajectory to center
  const xc = data.goal.distance / 2;
  const yc = (data.goal.height) / 2;

  const P = (px, py, pz = 0) => project(px - xc, py - yc, pz);

  // draw floor grid
  const gridSteps = 8;
  const gw = data.goal.distance * 1.3;
  const gh = gw * 0.5;
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= gridSteps; i++) {
    const t = i / gridSteps;
    const p0 = P(t * gw - gw * 0.1, 0, -gh / 2);
    const p1 = P(t * gw - gw * 0.1, 0,  gh / 2);
    ctx.beginPath(); ctx.moveTo(p0.sx, p0.sy); ctx.lineTo(p1.sx, p1.sy); ctx.stroke();
    const q0 = P(-gw * 0.1, 0, (t - 0.5) * gh);
    const q1 = P( gw * 0.9, 0, (t - 0.5) * gh);
    ctx.beginPath(); ctx.moveTo(q0.sx, q0.sy); ctx.lineTo(q1.sx, q1.sy); ctx.stroke();
  }

  // goal basket — U-shaped box open at top, correct depth & height
  const gd   = data.goal.depth;
  const gH   = data.goal.height;
  const gDist = data.goal.distance;
  // Half-width in z (visual only, ball is planar at z=0)
  const hw = 0.34;  // half of 26.5 in opening width
  // Front face corners (at goal_distance)
  const fBL = P(gDist,      0,  -hw), fBR = P(gDist,      0,   hw);
  const fTL = P(gDist,      gH, -hw), fTR = P(gDist,      gH,  hw);
  // Back face corners (at goal_distance + goal_depth)
  const bBL = P(gDist + gd, 0,  -hw), bBR = P(gDist + gd, 0,   hw);
  const bTL = P(gDist + gd, gH, -hw), bTR = P(gDist + gd, gH,  hw);

  const drawFace = (pts, fill, stroke) => {
    ctx.fillStyle = fill; ctx.strokeStyle = stroke; ctx.lineWidth = 1.5;
    ctx.beginPath();
    pts.forEach((pt, i) => i === 0 ? ctx.moveTo(pt.sx, pt.sy) : ctx.lineTo(pt.sx, pt.sy));
    ctx.closePath(); ctx.fill(); ctx.stroke();
  };
  // floor of basket
  drawFace([fBL, fBR, bBR, bBL], 'rgba(0,229,160,0.05)', 'rgba(0,229,160,0.3)');
  // front face
  drawFace([fBL, fBR, fTR, fTL], 'rgba(0,229,160,0.06)', 'rgba(0,229,160,0.5)');
  // back face
  drawFace([bBL, bBR, bTR, bTL], 'rgba(0,229,160,0.04)', 'rgba(0,229,160,0.3)');
  // top opening rim (just lines, no fill — it's open)
  ctx.strokeStyle = 'rgba(0,229,160,0.9)'; ctx.lineWidth = 2;
  ctx.beginPath();
  [fTL, fTR, bTR, bTL, fTL].forEach((pt, i) => i === 0 ? ctx.moveTo(pt.sx, pt.sy) : ctx.lineTo(pt.sx, pt.sy));
  ctx.stroke();

  // robot
  const rp = P(0, 0, 0);
  const rh = P(0, data.launch_height, 0);
  ctx.strokeStyle = '#2a3a4e'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(rp.sx, rp.sy); ctx.lineTo(rh.sx, rh.sy); ctx.stroke();
  ctx.fillStyle = '#00d4ff';
  ctx.beginPath(); ctx.arc(rh.sx, rh.sy, 4, 0, Math.PI * 2); ctx.fill();

  // trajectory — glow pass
  ctx.strokeStyle = 'rgba(0,212,255,0.12)'; ctx.lineWidth = 10; ctx.lineJoin = 'round';
  ctx.beginPath();
  data.x.forEach((xi, i) => {
    const pt = P(xi, data.y[i], 0);
    i === 0 ? ctx.moveTo(pt.sx, pt.sy) : ctx.lineTo(pt.sx, pt.sy);
  });
  ctx.stroke();

  // trajectory — main
  ctx.strokeStyle = '#00d4ff'; ctx.lineWidth = 2;
  ctx.beginPath();
  data.x.forEach((xi, i) => {
    const pt = P(xi, data.y[i], 0);
    i === 0 ? ctx.moveTo(pt.sx, pt.sy) : ctx.lineTo(pt.sx, pt.sy);
  });
  ctx.stroke();

  // shadow on floor
  ctx.strokeStyle = 'rgba(0,212,255,0.12)'; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
  ctx.beginPath();
  data.x.forEach((xi, i) => {
    const pt = P(xi, 0, 0);
    i === 0 ? ctx.moveTo(pt.sx, pt.sy) : ctx.lineTo(pt.sx, pt.sy);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  // apex
  const apx = P(data.apex_x, data.apex_y, 0);
  ctx.fillStyle = '#ffb547';
  ctx.shadowColor = '#ffb547'; ctx.shadowBlur = 8;
  ctx.beginPath(); ctx.arc(apx.sx, apx.sy, 5, 0, Math.PI * 2); ctx.fill();
  ctx.shadowBlur = 0;

  // impact dot — at (x_at_top, goal_height) on the top plane
  if (data.x_at_top != null) {
    const ip = P(data.x_at_top, data.goal.height, 0);
    ctx.fillStyle = data.made ? '#00e5a0' : '#ff4d6a';
    ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 10;
    ctx.beginPath(); ctx.arc(ip.sx, ip.sy, 7, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
  }

  ctx.fillStyle = '#00e5a0'; ctx.font = '10px JetBrains Mono'; ctx.textAlign = 'left';
  const goalLabel = P(data.goal.distance, data.goal.height + 0.10, 0);
  ctx.fillText(`GOAL top: ${data.goal.height.toFixed(2)} m`, goalLabel.sx + 6, goalLabel.sy);
}

// ══════════════════════════════════════════════════════════════════════════
// SHOT FAMILY  —  every shot that scores at this distance, drawn at once
// ══════════════════════════════════════════════════════════════════════════
// The tolerance sweep asks "how safe is THIS shot".  This asks the other
// question: which shots score at all, and which part of that family is safe.
// CLOUD mode answers a third — what YOUR shooter's scatter actually looks
// like, sampled rather than derived.

let lastFamily  = null;
let lastCloud   = null;
let familyMode  = 'family';        // 'family' | 'cloud'
let familySel   = -1;              // index of the shot loaded into the sliders
let familyHover = -1;
let familyHit   = null;            // hit-test snapshot: { shots, X, Y }

// Robustness → colour: on the edge of missing is red, comfortably inside is
// green, amber between.  The sidebar list and the arcs share this ramp so a
// row and its curve are the same colour.
//
// The top of the ramp is the best margin actually available, not a fixed 3σ.
// A long shot with a narrow band may top out under 1σ, and against a fixed
// scale every curve in it would be the same red — which is exactly when the
// picture needs to show WHICH of those marginal shots is least bad.
let marginFull = 1.0;

function setMarginScale(values) {
  const top = Math.max(0, ...values.filter(Number.isFinite));
  marginFull = Math.max(0.5, Math.ceil(top * 2) / 2);
}

function marginColor(normalizedMargin, alpha = 1) {
  const t = Math.max(0, Math.min(1, (normalizedMargin || 0) / marginFull));
  const stops = [[255, 77, 106], [255, 181, 71], [0, 229, 160]];
  const k = t < 0.5 ? 0 : 1;
  const f = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  const c = stops[k].map((z, i) => Math.round(z + (stops[k + 1][i] - z) * f));
  return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}

function drawFamily(data) {
  familyHit = null;
  if (!data || !data.shots || !data.shots.length) {
    emptyCanvasMessage(ui.familyCanvas, 'NO COMMANDABLE SHOT SCORES AT THIS DISTANCE');
    return;
  }

  setMarginScale(data.shots.map(s => s.margin_sigma));
  const { xMax, yMax } = worldExtents(data.shots, data.goal);
  const F = drawFieldFrame(ui.familyCanvas, { xMax, yMax, goal: data.goal });
  const { ctx, W, H, X, Y, padL, padR, padT, padB } = F;

  // Least robust first, so the safe shots end up on top of the fragile ones.
  const order = data.shots
    .map((s, i) => i)
    .sort((a, b) => data.shots[a].margin_sigma - data.shots[b].margin_sigma);

  ctx.lineWidth = 1.1;
  for (const i of order) {
    if (i === data.best_index || i === familySel || i === familyHover) continue;
    ctx.strokeStyle = marginColor(data.shots[i].margin_sigma, 0.5);
    strokePath(ctx, X, Y, data.shots[i]);
  }

  // The most robust shot in the family — what SOLVE OPTIMAL SHOT converges to.
  const best = data.shots[data.best_index];
  ctx.strokeStyle = marginColor(best.margin_sigma, 1);
  ctx.lineWidth = 2.2;
  ctx.shadowColor = marginColor(best.margin_sigma, 1);
  ctx.shadowBlur = 8;
  strokePath(ctx, X, Y, best);
  ctx.shadowBlur = 0;

  if (familyHover >= 0 && familyHover < data.shots.length) {
    ctx.strokeStyle = '#e8eef5'; ctx.lineWidth = 2;
    strokePath(ctx, X, Y, data.shots[familyHover]);
  }
  if (familySel >= 0 && familySel < data.shots.length) {
    ctx.strokeStyle = '#00d4ff'; ctx.lineWidth = 2.5;
    ctx.shadowColor = '#00d4ff'; ctx.shadowBlur = 8;
    strokePath(ctx, X, Y, data.shots[familySel]);
    ctx.shadowBlur = 0;
  }

  // Rim-crossing dots: where inside the opening each shot actually lands.
  const gy = Y(data.goal.height);
  const w  = data.goal.depth * (F.wPx / xMax);
  for (const s of data.shots) {
    if (s.entry_pos == null) continue;
    ctx.fillStyle = marginColor(s.margin_sigma, 0.85);
    ctx.beginPath();
    ctx.arc(F.gx + s.entry_pos * w, gy, 2, 0, Math.PI * 2);
    ctx.fill();
  }

  // ── legend + colour scale ────────────────────────────────────────────────
  const lx = padL + 10, ly = padT + 10;
  ctx.font = '10px JetBrains Mono'; ctx.textAlign = 'left';
  ctx.fillStyle = '#b4c2d0';
  ctx.fillText(`${data.count} SCORING SHOTS  ·  ${data.angle_span[0]}–${data.angle_span[1]}°`, lx, ly + 4);

  const barW = 120, barH = 6, by = ly + 14;
  for (let i = 0; i < barW; i++) {
    ctx.fillStyle = marginColor(i / (barW - 1) * marginFull, 1);
    ctx.fillRect(lx + i, by, 1, barH);
  }
  ctx.fillStyle = '#637587'; ctx.font = '9px JetBrains Mono';
  ctx.fillText('0×', lx, by + barH + 10);
  ctx.textAlign = 'right';
  ctx.fillText(`${marginFull.toFixed(1)}× HALF-WIDTH MARGIN`, lx + barW, by + barH + 10);
  ctx.textAlign = 'left';
  ctx.fillStyle = '#b4c2d0'; ctx.font = '10px JetBrains Mono';
  ctx.fillText(`BEST  ${best.angle_deg.toFixed(1)}°  ${best.velocity.toFixed(2)} m/s  ` +
               `${best.margin_sigma.toFixed(2)}× half-width`, lx, by + barH + 26);

  ctx.fillStyle = '#637587'; ctx.font = '9px JetBrains Mono';
  ctx.textAlign = 'right';
  ctx.fillText('DISTANCE (m) →', W - padR, H - padB + 28);
  ctx.textAlign = 'left';

  familyHit = { shots: data.shots, X, Y };
}

// ── Monte-Carlo cloud ───────────────────────────────────────────────────────
function drawCloud(data) {
  familyHit = null;
  if (!data || !data.samples || !data.samples.length) {
    emptyCanvasMessage(ui.familyCanvas, 'NO SAMPLES');
    return;
  }

  const { xMax, yMax } = worldExtents(data.samples, data.goal);
  const F = drawFieldFrame(ui.familyCanvas, { xMax, yMax, goal: data.goal });
  const { ctx, W, H, X, Y, padL, padR, padT, padB } = F;

  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(255,77,106,0.16)';
  for (const s of data.samples) if (!s.made) strokePath(ctx, X, Y, s);
  ctx.strokeStyle = 'rgba(0,229,160,0.30)';
  for (const s of data.samples) if (s.made) strokePath(ctx, X, Y, s);

  // ── empirical vs analytic ────────────────────────────────────────────────
  // Two independent routes to the same probability.  If they disagree by more
  // than a few standard errors, one of them is wrong.
  const emp = data.p_empirical * 100;
  const reference = data.coverage_reference * 100;
  const se  = data.p_stderr    * 100;
  const lx = padL + 10, ly = padT + 10;
  ctx.font = '10px JetBrains Mono'; ctx.textAlign = 'left';
  ctx.fillStyle = '#b4c2d0';
  ctx.fillText(`${data.hits}/${data.n} SCORED`, lx, ly + 4);
  ctx.fillStyle = '#00e5a0';
  ctx.fillText(`SAMPLED    ${emp.toFixed(1)}% ±${se.toFixed(1)}`, lx, ly + 18);
  ctx.fillStyle = '#00d4ff';
  ctx.fillText(`BOX GRID   ${reference.toFixed(1)}%`, lx, ly + 32);
  ctx.fillStyle = '#637587'; ctx.font = '9px JetBrains Mono';
  ctx.fillText(`±v ${data.half_width.v.toFixed(2)} m/s  ·  ±a ${data.half_width.a.toFixed(1)}°`, lx, ly + 46);
  ctx.textAlign = 'right';
  ctx.fillText('DISTANCE (m) →', W - padR, H - padB + 28);
  ctx.textAlign = 'left';
}

function drawFamilyTab() {
  if (familyMode === 'cloud') {
    if (!lastCloud) return emptyCanvasMessage(ui.familyCanvas, 'RUN TO SAMPLE YOUR SHOOTER\u2019S SCATTER');
    drawCloud(lastCloud);
  } else {
    if (!lastFamily) return emptyCanvasMessage(ui.familyCanvas, 'RUN SHOT FAMILY TO POPULATE THIS VIEW');
    drawFamily(lastFamily);
  }
}

// ── ALL SHOTS list ──────────────────────────────────────────────────────────
function renderFamilyList(data) {
  ui.familyList.innerHTML = '';
  if (!data || !data.shots.length) return;
  setMarginScale(data.shots.map(s => s.margin_sigma));

  data.shots.forEach((s, i) => {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'family-row';
    row.style.setProperty('--row', marginColor(s.margin_sigma, 1));
    row.classList.toggle('selected', i === familySel);
    row.setAttribute('aria-pressed', String(i === familySel));
    row.innerHTML =
      `<span class="fr-a">${s.angle_deg.toFixed(0)}°</span>` +
      `<span class="fr-v">${s.velocity.toFixed(2)} m/s</span>` +
      `<span class="fr-p">${(s.p_make * 100).toFixed(0)}%</span>`;
    row.title = `${s.margin_sigma.toFixed(2)}× half-width margin · entry ${(s.entry_pos * 100).toFixed(0)}% · ` +
                `${s.flight_time.toFixed(2)} s`;
    row.addEventListener('click', () => selectFamilyShot(i));
    ui.familyList.appendChild(row);
  });
}

function selectFamilyShot(i) {
  const data = lastFamily;
  if (!data || i < 0 || i >= data.shots.length) return;
  const s = data.shots[i];
  familySel = i;

  ui.velocity.value = s.velocity.toFixed(2);
  ui.angle.value    = s.angle_deg.toFixed(1);
  updateLabels();
  [...ui.familyList.children].forEach((row, k) => {
    row.classList.toggle('selected', k === i);
    row.setAttribute('aria-pressed', String(k === i));
  });
  ui.familyList.children[i]?.scrollIntoView({ block: 'nearest' });

  // The sliders quantise (0.05 m/s, 0.1°), so re-read them rather than trust
  // the family value — what is drawn must be what the readouts say.
  const vSnap = +ui.velocity.value, aSnap = +ui.angle.value;
  setFamilyStatus(
    `${aSnap.toFixed(1)}° · ${vSnap.toFixed(2)} m/s loaded · ` +
    `${s.margin_sigma.toFixed(2)}× half-width margin · coverage ${(s.coverage * 100).toFixed(0)}% · ` +
    `entry ${(s.entry_pos * 100).toFixed(0)}% · ${s.flight_time.toFixed(2)} s`,
    'success');
  if (familyMode === 'family') drawFamily(data);
  refreshTraj();
}

function setFamilyStatus(message, kind = '') {
  ui.familyStatus.textContent = message;
  ui.familyStatus.className = `status-line${kind ? ' ' + kind : ''}`;
}

// ── Hover / click on the fan ────────────────────────────────────────────────
const PICK_RADIUS_PX = 12;

function pickFamilyShot(px, py) {
  if (!familyHit) return -1;
  const { shots, X, Y } = familyHit;
  let best = -1, bestD = PICK_RADIUS_PX * PICK_RADIUS_PX;
  for (let i = 0; i < shots.length; i++) {
    const s = shots[i];
    for (let k = 0; k < s.x.length; k++) {
      const dx = X(s.x[k]) - px, dy = Y(s.y[k]) - py;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    }
  }
  return best;
}

let familyTipFrame = null;

function onFamilyMove(ev) {
  if (familyMode !== 'family' || !familyHit) return;
  const rect = ui.familyCanvas.getBoundingClientRect();
  const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
  if (familyTipFrame) return;
  familyTipFrame = requestAnimationFrame(() => {
    familyTipFrame = null;
    const hit = pickFamilyShot(px, py);
    if (hit !== familyHover) {
      familyHover = hit;
      drawFamily(lastFamily);
    }
    if (hit < 0) { ui.familyTip.classList.add('hidden'); return; }
    const s = lastFamily.shots[hit];
    ui.familyTip.innerHTML =
      `<strong>${s.angle_deg.toFixed(1)}° · ${s.velocity.toFixed(2)} m/s</strong>` +
      `<span>MARGIN ${s.margin_sigma.toFixed(2)}× half-width · COVERAGE ${(s.coverage * 100).toFixed(1)}%</span>` +
      `<span>SPEED WINDOW ${s.v_lo.toFixed(2)}–${s.v_hi.toFixed(2)} m/s</span>` +
      `<span>ENTRY ${(s.entry_pos * 100).toFixed(0)}% (${s.entry_pos > 0.5 ? 'BACK' : 'FRONT'}) ` +
      `@ ${s.entry_angle_deg == null ? '—' : s.entry_angle_deg.toFixed(0) + '°'}</span>` +
      `<span>APEX ${s.apex_y.toFixed(2)} m · ${s.flight_time.toFixed(2)} s</span>`;
    ui.familyTip.classList.remove('hidden');
    const tw = ui.familyTip.offsetWidth, th = ui.familyTip.offsetHeight;
    ui.familyTip.style.left = Math.min(px + 14, rect.width  - tw - 6) + 'px';
    ui.familyTip.style.top  = Math.min(py + 14, rect.height - th - 6) + 'px';
  });
}

function onFamilyLeave() {
  ui.familyTip.classList.add('hidden');
  if (familyHover !== -1) { familyHover = -1; if (familyMode === 'family') drawFamily(lastFamily); }
}

function onFamilyClick(ev) {
  if (familyMode !== 'family' || !familyHit) return;
  const rect = ui.familyCanvas.getBoundingClientRect();
  const hit = pickFamilyShot(ev.clientX - rect.left, ev.clientY - rect.top);
  if (hit >= 0) selectFamilyShot(hit);
}

// ══════════════════════════════════════════════════════════════════════════
// TARGET SWEEP  —  the most robust shot at every distance
// ══════════════════════════════════════════════════════════════════════════

let lastTargets = null;

function drawTargets(rows) {
  if (!rows || !rows.length) {
    emptyCanvasMessage(ui.targetsCanvas, 'RUN THE TARGET SWEEP TO POPULATE THIS VIEW');
    return;
  }
  const { ctx, W, H } = fitCanvas(ui.targetsCanvas);
  ctx.clearRect(0, 0, W, H);

  const padL = 54, padR = 54, padT = 26, padB = 56;
  const wPx = W - padL - padR;
  const hPx = H - padT - padB;

  const dIn = rows.map(r => r.distance_m / IN2M);
  const d0 = dIn[0], d1 = dIn[dIn.length - 1] || d0 + 1;
  const solved = rows.filter(r => r.ok);
  // Round the axis up to a whole 2 m/s so the ticks land on round numbers.
  const vTick = 2;
  const vMax  = Math.ceil(Math.max(2, ...solved.map(r => r.velocity)) * 1.08 / vTick) * vTick;
  setMarginScale(solved.map(r => r.margin_sigma));

  const X = d => padL + (d - d0) / Math.max(d1 - d0, 1e-6) * wPx;
  const YV = v => H - padB - v / vMax * hPx;                 // left: m/s
  const YA = a => H - padB - (a - 20) / 60 * hPx;            // right: 20–80°

  // grid + frame
  ctx.strokeStyle = 'rgba(255,255,255,0.04)'; ctx.lineWidth = 1;
  for (let v = 0; v <= vMax; v += vTick) {
    ctx.beginPath(); ctx.moveTo(padL, YV(v)); ctx.lineTo(W - padR, YV(v)); ctx.stroke();
  }
  ctx.strokeStyle = '#2a3a4e'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB); ctx.lineTo(W - padR, H - padB);
  ctx.stroke();
  ctx.beginPath(); ctx.moveTo(W - padR, padT); ctx.lineTo(W - padR, H - padB); ctx.stroke();

  // ticks
  ctx.font = '10px JetBrains Mono'; ctx.fillStyle = '#637587';
  ctx.textAlign = 'center';
  const dStep = Math.max(6, Math.round((d1 - d0) / 8 / 6) * 6);
  for (let d = Math.ceil(d0 / dStep) * dStep; d <= d1; d += dStep) {
    ctx.fillText(d.toFixed(0) + '"', X(d), H - padB + 14);
  }
  ctx.textAlign = 'right';
  for (let v = 0; v <= vMax; v += vTick) ctx.fillText(v.toFixed(0), padL - 5, YV(v) + 3);
  ctx.textAlign = 'left';
  for (let a = 20; a <= 80; a += 10) ctx.fillText(a + '°', W - padR + 6, YA(a) + 3);

  // Broken into runs so an unreachable stretch shows as a gap, not a chord.
  const runs = [];
  let run = [];
  for (const r of rows) { if (r.ok) run.push(r); else if (run.length) { runs.push(run); run = []; } }
  if (run.length) runs.push(run);

  const line = (key, proj, color, width) => {
    ctx.strokeStyle = color; ctx.lineWidth = width;
    for (const seg of runs) {
      ctx.beginPath();
      seg.forEach((r, i) => {
        const px = X(r.distance_m / IN2M), py = proj(r[key]);
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      });
      ctx.stroke();
    }
  };
  line('angle_deg', YA, '#ffb547', 1.6);
  line('velocity',  YV, '#00d4ff', 2.0);

  // margin strip along the bottom — where in the range the shot is comfortable
  const stripY = H - padB + 22, stripH = 8;
  const bw = Math.max(2, wPx / Math.max(rows.length, 1));
  for (const r of rows) {
    const px = X(r.distance_m / IN2M) - bw / 2;
    ctx.fillStyle = r.ok ? marginColor(r.margin_sigma, 0.9) : 'rgba(58,74,92,0.5)';
    ctx.fillRect(px, stripY, bw, stripH);
  }
  ctx.fillStyle = '#637587'; ctx.font = '9px JetBrains Mono';
  ctx.textAlign = 'left';
  ctx.fillText(`MARGIN  0–${marginFull.toFixed(1)}× half-width`, padL, stripY + stripH + 10);
  ctx.textAlign = 'right';
  ctx.fillText('GOAL DISTANCE →', W - padR, stripY + stripH + 10);

  // legend
  ctx.textAlign = 'left'; ctx.font = '10px JetBrains Mono';
  const lx = padL + 10, ly = padT + 12;
  ctx.fillStyle = '#00d4ff'; ctx.fillRect(lx, ly, 14, 2);
  ctx.fillStyle = '#b4c2d0'; ctx.fillText('EXIT VELOCITY (m/s, left)', lx + 18, ly + 4);
  ctx.fillStyle = '#ffb547'; ctx.fillRect(lx, ly + 14, 14, 2);
  ctx.fillStyle = '#b4c2d0'; ctx.fillText('LAUNCH ANGLE (°, right)', lx + 18, ly + 18);
}

function targetSweepCSV(rows) {
  const cols = ['distance_m', 'distance_in', 'ok', 'velocity', 'angle_deg',
                'window_pct', 'margin_sigma', 'p_make', 'v_down_ms', 'v_up_ms',
                'a_down_deg', 'a_up_deg', 'entry_angle_deg', 'entry_pos',
                'flight_time', 'apex_y'];
  const cell = z => (z == null ? '' : typeof z === 'number' ? z : String(z));
  const lines = [cols.join(',')];
  for (const r of rows) {
    const row = { ...r, distance_in: +(r.distance_m / IN2M).toFixed(3), ok: r.ok ? 1 : 0 };
    lines.push(cols.map(c => cell(row[c])).join(','));
  }
  return lines.join('\n') + '\n';
}

// ══════════════════════════════════════════════════════════════════════════
// REFRESH LOGIC
// ══════════════════════════════════════════════════════════════════════════

let trajTimer = null;

function scheduleTraj() {
  clearTimeout(trajTimer);
  trajTimer = setTimeout(refreshTraj, 40);
}

async function refreshTraj(overrides = {}) {
  const p = { ...readParams(), ...overrides };
  const data = await API('simulate', p);
  lastTraj = data;
  showTrajStats(data);
  if (activeTab === 'side')  drawTrajectory(data);
  if (activeTab === 'iso')   draw3D(data);
}

async function refreshSweep() {
  const p = readParams();
  const dv = +ui.sweepDv.value;
  const da = +ui.sweepDa.value;
  const n  = +ui.resolution.value;
  ui.btnWin.textContent = '⟳ COMPUTING…';
  ui.btnWin.disabled = true;
  try {
    const data = await API('sweep', {
      ...p,
      dv_range: dv,
      da_range: da,
      resolution: n,
    });
    lastSweep = data;
    // land on the sweep tab unless the user is already on a sweep-fed view
    if (activeTab !== 'sweep' && activeTab !== 'tolmap') showTab('sweep');
    else                                                 redrawTab(activeTab);
  } finally {
    ui.btnWin.textContent = '⟳ RUN SWEEP';
    ui.btnWin.disabled = false;
  }
}

async function runFamily() {
  const p  = readParams();
  const dv = +ui.sweepDv.value;
  const da = +ui.sweepDa.value;
  const cloud = familyMode === 'cloud';
  const label = ui.btnFamily.textContent;
  ui.btnFamily.textContent = cloud ? '◠ SAMPLING…' : '◠ SOLVING FAMILY…';
  ui.btnFamily.disabled = true;
  setFamilyStatus(cloud ? 'Firing samples…' : 'Solving the scoring band…', 'loading');
  try {
    if (cloud) {
      lastCloud = await API('montecarlo', {
        ...p, dv_range: dv, da_range: da, samples: +ui.mcSamples.value,
      });
      setFamilyStatus(
        `${lastCloud.hits}/${lastCloud.n} scored — sampled ` +
        `${(lastCloud.p_empirical * 100).toFixed(1)}% ` +
        `±${(lastCloud.p_stderr * 100).toFixed(1)} vs box-grid ` +
        `${(lastCloud.coverage_reference * 100).toFixed(1)}%.`, 'success');
    } else {
      lastFamily = await API('family', {
        ...p, dv_range: dv, da_range: da, angle_step: +ui.familyStep.value,
      });
      familySel = -1;
      familyHover = -1;
      renderFamilyList(lastFamily);
      if (!lastFamily.count) {
        setFamilyStatus('No commandable shot scores at this distance.', 'warning');
      } else {
        const b = lastFamily.shots[lastFamily.best_index];
        setFamilyStatus(
          `${lastFamily.count} scoring shots across ${lastFamily.angle_span[0]}–` +
          `${lastFamily.angle_span[1]}°. Most robust: ${b.angle_deg.toFixed(1)}° at ` +
          `${b.velocity.toFixed(2)} m/s (${b.window_pct.toFixed(0)}% make window).`, 'success');
      }
    }
    showTab('family');
  } catch (err) {
    setFamilyStatus(`Failed: ${err.message}`, 'error');
  } finally {
    ui.btnFamily.textContent = label;
    ui.btnFamily.disabled = false;
  }
}

// Distances go up in chunks so the progress bar reflects real completed work.
// The endpoint is stateless, so this needs no job tracking on either side.
const TS_CHUNK = 8;
const TS_MAX_POINTS = 200;

function setTargetStatus(message, kind = '') {
  ui.tsStatus.textContent = message;
  ui.tsStatus.className = `status-line${kind ? ' ' + kind : ''}`;
}

function setTargetProgress(done, total) {
  ui.tsFill.style.width = (total ? done / total * 100 : 0).toFixed(1) + '%';
}

async function runTargetSweep() {
  const from = +ui.tsFrom.value, to = +ui.tsTo.value, step = +ui.tsStep.value;
  if (![from, to, step].every(Number.isFinite) || step <= 0 || to < from || from <= 0) {
    setTargetStatus('Enter a valid from / to / step range.', 'error');
    return;
  }

  const distances = [];
  for (let d = from; d <= to + 1e-9 && distances.length < TS_MAX_POINTS; d += step) {
    distances.push(d * IN2M);
  }
  if (distances.length >= TS_MAX_POINTS) {
    setTargetStatus(`Capped at ${TS_MAX_POINTS} distances — widen the step.`, 'warning');
  }

  const body = { ...readParams(), dv_range: +ui.sweepDv.value, da_range: +ui.sweepDa.value };
  const label = ui.btnTargets.textContent;
  ui.btnTargets.textContent = '⇉ SWEEPING…';
  ui.btnTargets.disabled = true;
  ui.btnTargetsCsv.disabled = true;
  setTargetProgress(0, distances.length);

  const rows = [];
  try {
    for (let i = 0; i < distances.length; i += TS_CHUNK) {
      const r = await API('target_sweep', {
        ...body, distances: distances.slice(i, i + TS_CHUNK),
      });
      if (!r.ok) throw new Error(r.error || 'sweep failed');
      rows.push(...r.rows);
      lastTargets = rows.slice();
      setTargetProgress(rows.length, distances.length);
      setTargetStatus(`${rows.length} / ${distances.length} distances solved…`, 'loading');
      if (activeTab === 'targets') drawTargets(lastTargets);
    }
    const solved = rows.filter(r => r.ok);
    const best = solved.reduce((a, r) => (a && a.margin_sigma >= r.margin_sigma ? a : r), null);
    setTargetStatus(
      `${solved.length} / ${rows.length} distances reachable` +
      (best ? `. Widest normalized margin ${best.margin_sigma.toFixed(2)} at ` +
              `${(best.distance_m / IN2M).toFixed(0)}".` : '.'),
      solved.length ? 'success' : 'warning');
    ui.btnTargetsCsv.disabled = solved.length === 0;
    showTab('targets');
  } catch (err) {
    setTargetStatus(`Failed: ${err.message}`, 'error');
  } finally {
    ui.btnTargets.textContent = label;
    ui.btnTargets.disabled = false;
  }
}

function downloadTargetSweep() {
  if (!lastTargets || !lastTargets.length) return;
  const p = readParams();
  const blob = new Blob([targetSweepCSV(lastTargets)], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `decode_target_sweep_h${(p.launch_height * 100).toFixed(0)}cm` +
               `_drag${p.enable_drag ? 'on' : 'off'}.csv`;
  document.body.appendChild(a); a.click();
  a.remove(); URL.revokeObjectURL(url);
}

async function runOptimize() {
  const p = readParams();
  ui.btnOpt.textContent = '⟶ SOLVING…';
  ui.btnOpt.disabled = true;
  try {
    const r = await API('optimize', {
      goal_distance: p.goal_distance,
      launch_height: p.launch_height,
      wind:          p.wind,
      enable_drag:   p.enable_drag,
      dv_range:      +ui.sweepDv.value,
      da_range:      +ui.sweepDa.value,
    });
    if (r.ok) {
      ui.velocity.value = r.velocity.toFixed(2);
      ui.angle.value    = r.angle_deg.toFixed(1);
      updateLabels();
      ui.optOut.classList.add('show');
      const m = r.margins || {};
      const num = (z, dp) => (z == null ? '—' : z.toFixed(dp));
      ui.optOut.innerHTML = `
        <span class="key">OPTIMAL VELOCITY</span><span class="v">${r.velocity.toFixed(2)} m/s</span>
        <span class="key">OPTIMAL ANGLE</span><span class="v">${r.angle_deg.toFixed(1)} °</span>
        <span class="key">MAKE WINDOW</span><span class="v">${r.window_pct.toFixed(1)} %</span>
        <span class="key">ROBUST COVERAGE</span><span class="v">${(r.coverage * 100).toFixed(1)} %</span>
        <span class="key">NORMALIZED MARGIN</span><span class="v">${r.margin_sigma.toFixed(2)} × half-width</span>
        <span class="key">VELOCITY SLACK</span><span class="v">−${num(m.v_down_ms, 2)} / +${num(m.v_up_ms, 2)} m/s</span>
        <span class="key">ANGLE SLACK</span><span class="v">−${num(m.a_down_deg, 1)} / +${num(m.a_up_deg, 1)} °</span>
      `;
      await refreshTraj();
    }
  } finally {
    ui.btnOpt.textContent = '⟶ SOLVE OPTIMAL SHOT';
    ui.btnOpt.disabled = false;
  }
}

async function downloadLUT() {
  const p = readParams();
  ui.btnDownload.textContent = '⬇ GENERATING…';
  ui.btnDownload.disabled = true;
  try {
    const resp = await fetch('/api/lut', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(p),
    });
    const blob = await resp.blob();
    const disposition = resp.headers.get('Content-Disposition') || '';
    const match = disposition.match(/filename="?([^"]+)"?/);
    const filename = match ? match[1] : 'decode_lut.csv';
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    a.remove(); URL.revokeObjectURL(url);
  } finally {
    ui.btnDownload.textContent = '⬇ DOWNLOAD LOOKUP TABLE';
    ui.btnDownload.disabled = false;
  }
}

// ── Event binding ──────────────────────────────────────────────────────────

[ui.velocity, ui.angle, ui.height, ui.distance, ui.wind].forEach(el => {
  el.addEventListener('input', () => { updateLabels(); scheduleTraj(); });
});
[ui.sweepDv, ui.sweepDa].forEach(el => {
  el.addEventListener('input', () => { updateLabels(); });
});
ui.drag.addEventListener('change', () => scheduleTraj());
ui.btnWin.addEventListener('click', refreshSweep);
ui.btnOpt.addEventListener('click', runOptimize);
ui.btnDownload.addEventListener('click', downloadLUT);

ui.btnFamily.addEventListener('click', runFamily);
ui.btnTargets.addEventListener('click', runTargetSweep);
ui.btnTargetsCsv.addEventListener('click', downloadTargetSweep);

ui.familyModeCtl.querySelectorAll('.seg').forEach(btn => {
  btn.addEventListener('click', () => {
    familyMode = btn.dataset.mode;
    ui.familyModeCtl.querySelectorAll('.seg').forEach(b => {
      b.classList.toggle('active', b === btn);
      b.setAttribute('aria-pressed', String(b === btn));
    });
    ui.familyTip.classList.add('hidden');
    ui.familyStep.parentElement.classList.toggle('hidden', familyMode === 'cloud');
    ui.mcSamples.parentElement.classList.toggle('hidden', familyMode !== 'cloud');
    if (activeTab === 'family') drawFamilyTab();
  });
});

ui.familyCanvas.addEventListener('mousemove', onFamilyMove);
ui.familyCanvas.addEventListener('mouseleave', onFamilyLeave);
ui.familyCanvas.addEventListener('click', onFamilyClick);

[ui.az, ui.el].forEach(el => {
  el.addEventListener('input', () => {
    setSliderFill(el);
    if (lastTraj) draw3D(lastTraj);
  });
  setSliderFill(el);
});

window.addEventListener('resize', () => redrawTab(activeTab));

// ── Boot ──────────────────────────────────────────────────────────────────
updateLabels();
refreshTraj();
