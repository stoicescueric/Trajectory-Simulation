/* DECODE Shot Simulator — App Logic */

const API = (path, body) =>
  fetch(`/api/${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(r => r.json());

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
  rdApex: $('rd-apex'), rdTime: $('rd-time'),
  rdImpact: $('rd-impact'), rdEntry: $('rd-entry'),
  rdMade: $('rd-made'), rdWindow: $('rd-window'),
  resultReadout: $('result-readout'),
  btnOpt: $('btn-optimize'), btnWin: $('btn-window'),
  btnDownload: $('btn-download'),
  optOut: $('optimize-out'),
  az: $('az'), el: $('el'),
  tabs: document.querySelectorAll('.tab'),
  tabPanels: { side: $('tab-side'), iso: $('tab-iso'), sweep: $('tab-sweep') },
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
  ui.dvd.innerHTML = `±${dv.toFixed(1)} <em>m/s</em>`;
  setSliderFill(ui.sweepDv);

  const da = +ui.sweepDa.value;
  ui.dad.innerHTML = `±${da.toFixed(1)} <em>°</em>`;
  setSliderFill(ui.sweepDa);
}

// ── Tab switching ──────────────────────────────────────────────────────────

let activeTab = 'side';

ui.tabs.forEach(btn => {
  btn.addEventListener('click', () => {
    ui.tabs.forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    const name = btn.dataset.tab;
    activeTab = name;
    Object.entries(ui.tabPanels).forEach(([k, el]) => {
      el.classList.toggle('hidden', k !== name);
    });
    if (name === 'iso' && lastTraj) draw3D(lastTraj);
    if (name === 'sweep' && lastSweep) drawSweep(lastSweep);
    else if (name === 'side' && lastTraj) drawTrajectory(lastTraj);
  });
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

  // readouts
  ui.rdApex.textContent   = `${data.apex_y.toFixed(2)} m`;
  ui.rdTime.textContent   = `${data.t[data.t.length - 1].toFixed(3)} s`;
  // "Y @ GOAL" now shows height at front face; entry shows angle at top crossing
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
// SPAGHETTI SWEEP CANVAS
// ══════════════════════════════════════════════════════════════════════════

let lastSweep = null;

function drawSweep(sweepData) {
  const { ctx, W, H } = fitCanvas(ui.sweepCanvas);
  ctx.clearRect(0, 0, W, H);

  const padL = 54, padR = 30, padT = 28, padB = 38;
  const trajs = sweepData.trajectories;
  if (!trajs || !trajs.length) return;

  // world extents across all trajectories
  let xMax = 0, yMax = 0;
  for (const tr of trajs) {
    for (const xi of tr.x) xMax = Math.max(xMax, xi);
    for (const yi of tr.y) yMax = Math.max(yMax, yi);
  }
  xMax = Math.max(xMax, sweepData.goal.distance + sweepData.goal.depth + 0.3) + 0.1;
  yMax = Math.max(yMax, sweepData.goal.height + 0.5) + 0.1;

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
  const gx  = X(sweepData.goal.distance);
  const gy  = Y(sweepData.goal.height);
  const gd  = sweepData.goal.depth * (wPx / xMax);
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

  const misses = trajs.filter(t => !t.made);
  const makes  = trajs.filter(t =>  t.made);

  drawGroup(misses, '#ff4d6a', 0.18, 1.0);
  drawGroup(makes,  '#00e5a0', 0.65, 1.5);

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
  ctx.fillStyle = '#00d4ff'; ctx.fillRect(lx, ly + 28, 14, 2);
  ctx.fillStyle = '#b4c2d0';
  ctx.fillText('NOMINAL', lx + 18, ly + 32);

  // make %
  const makePct = trajs.length ? (makes.length / trajs.length * 100).toFixed(1) : '—';
  ui.rdWindow.textContent = makePct + '%';

  // axes labels
  ctx.fillStyle = '#637587'; ctx.font = '9px JetBrains Mono';
  ctx.textAlign = 'right';
  ctx.fillText('DISTANCE (m) →', W - padR, H - padB + 28);
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
// REFRESH LOGIC
// ══════════════════════════════════════════════════════════════════════════

let trajTimer = null;

function scheduleTraj() {
  clearTimeout(trajTimer);
  trajTimer = setTimeout(refreshTraj, 40);
}

async function refreshTraj() {
  const p = readParams();
  const data = await API('simulate', p);
  lastTraj = data;
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
    if (activeTab !== 'sweep') {
      // switch to sweep tab
      ui.tabs.forEach(t => t.classList.remove('active'));
      document.querySelector('[data-tab="sweep"]').classList.add('active');
      Object.entries(ui.tabPanels).forEach(([k, el]) => el.classList.toggle('hidden', k !== 'sweep'));
      activeTab = 'sweep';
    }
    drawSweep(data);
  } finally {
    ui.btnWin.textContent = '⟳ RUN SWEEP';
    ui.btnWin.disabled = false;
  }
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
      ui.optOut.innerHTML = `
        <span class="key">OPTIMAL VELOCITY</span><span class="v">${r.velocity.toFixed(2)} m/s</span>
        <span class="key">OPTIMAL ANGLE</span><span class="v">${r.angle_deg.toFixed(1)} °</span>
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

[ui.az, ui.el].forEach(el => {
  el.addEventListener('input', () => {
    setSliderFill(el);
    if (lastTraj) draw3D(lastTraj);
  });
  setSliderFill(el);
});

window.addEventListener('resize', () => {
  if (activeTab === 'side'  && lastTraj)  drawTrajectory(lastTraj);
  if (activeTab === 'iso'   && lastTraj)  draw3D(lastTraj);
  if (activeTab === 'sweep' && lastSweep) drawSweep(lastSweep);
});

// ── Boot ──────────────────────────────────────────────────────────────────
updateLabels();
refreshTraj();
