"""
Microplastic Detection System(ESP32-CAM + YOLOv8) — Integration Script
Real-time lensless digital holography + YOLOv8 inference pipeline.

Usage:
    python 03_esp32_integration.py
    python 03_esp32_integration.py --esp32 192.168.29.9 --model yolov8_microplastic_trained.pt
    python 03_esp32_integration.py --esp32 192.168.29.9 --conf 0.3 --port 5000
"""

import cv2
from ultralytics import YOLO
from flask import Flask, Response, jsonify, render_template_string, request, send_file
import argparse
import time
import threading
import collections
import base64
import io
from pathlib import Path

# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Microplastic Detection Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #060a0e;
  --surface:  #0b1017;
  --card:     #0f171f;
  --border:   #182433;
  --accent:   #00e5ff;
  --green:    #00ff99;
  --warn:     #ffb800;
  --danger:   #ff3d3d;
  --txt:      #c0d4e4;
  --dim:      #3a5060;
  --mono:     'Share Tech Mono', monospace;
  --sans:     'Syne', sans-serif;
  --r:        6px;
  --glow:     0 0 20px rgba(0,229,255,0.2);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--txt);
  font-family: var(--sans);
  min-height: 100vh;
  overflow-x: hidden;
}

/* Dot-grid bg */
body::before {
  content: '';
  position: fixed; inset: 0;
  background-image: radial-gradient(rgba(0,229,255,0.06) 1px, transparent 1px);
  background-size: 28px 28px;
  pointer-events: none;
  z-index: 0;
}

/* ── Header ─────────────────── */
header {
  position: relative; z-index: 10;
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 28px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
}
.brand { display: flex; align-items: center; gap: 14px; }
.brand-icon {
  width: 40px; height: 40px;
  border: 2px solid var(--accent); border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px;
  animation: halo 3s ease-in-out infinite;
  box-shadow: var(--glow);
}
@keyframes halo {
  0%,100% { box-shadow: 0 0 10px rgba(0,229,255,0.2); }
  50%      { box-shadow: 0 0 30px rgba(0,229,255,0.6); }
}
.brand-name { font-size: 1.3rem; font-weight: 800; letter-spacing: .04em; }
.brand-name span { color: var(--accent); }
.brand-sub { font-family: var(--mono); font-size: .62rem; color: var(--dim); margin-top: 2px; }

.header-right { display: flex; align-items: center; gap: 18px; }
.pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: 999px;
  border: 1px solid var(--border);
  font-family: var(--mono); font-size: .68rem; color: var(--dim);
}
.pill.online  { border-color: var(--green); color: var(--green); }
.pill.offline { border-color: var(--danger); color: var(--danger); }
.dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: currentColor;
  animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.15} }

/* ── Main grid ──────────────── */
main {
  position: relative; z-index: 1;
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: 18px;
  padding: 20px 28px;
  max-width: 1500px; margin: 0 auto;
}
@media(max-width:900px){ main{ grid-template-columns:1fr; padding:14px; } }

/* ── Card ───────────────────── */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--r);
  overflow: hidden;
  animation: rise .5s ease both;
}
@keyframes rise { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:none} }
.card:nth-child(1){animation-delay:.05s}
.card:nth-child(2){animation-delay:.10s}
.card:nth-child(3){animation-delay:.15s}
.card:nth-child(4){animation-delay:.20s}

.card-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 11px 16px;
  border-bottom: 1px solid var(--border);
  background: rgba(0,229,255,.03);
}
.card-title {
  font-family: var(--mono); font-size: .7rem;
  letter-spacing: .12em; text-transform: uppercase; color: var(--accent);
}
.card-body { padding: 16px; }

/* ── Video ──────────────────── */
.video-wrap {
  position: relative;
  background: #000;
  aspect-ratio: 4/3;
  overflow: hidden;
}
.video-wrap::after {
  content: '';
  position: absolute; inset: 0;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 3px,
    rgba(0,0,0,.07) 3px, rgba(0,0,0,.07) 4px
  );
  pointer-events: none; z-index: 2;
}
#feed { width: 100%; height: 100%; object-fit: cover; display: block; }
.vid-overlay {
  position: absolute; z-index: 3; pointer-events: none;
  font-family: var(--mono); font-size: .65rem;
  text-shadow: 0 0 8px rgba(0,229,255,.7);
}
.vid-tl { top:10px; left:12px; color: var(--accent); }
.vid-tr { top:10px; right:12px; color: var(--green); }
.vid-bl { bottom:10px; left:12px; color: rgba(0,229,255,.5); }
.vid-br { bottom:10px; right:12px; color: var(--warn); }

/* ── Stat row ───────────────── */
.stat-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  padding: 12px 16px;
  border-top: 1px solid var(--border);
}
.stat { text-align: center; }
.stat-val {
  font-family: var(--mono); font-size: 1.5rem; font-weight: bold;
  color: var(--accent); line-height: 1;
}
.stat-val.green  { color: var(--green); }
.stat-val.warn   { color: var(--warn); }
.stat-val.danger { color: var(--danger); }
.stat-lbl {
  font-family: var(--mono); font-size: .6rem;
  color: var(--dim); margin-top: 4px; letter-spacing: .08em;
}

/* ── Right column ───────────── */
.right { display: flex; flex-direction: column; gap: 14px; }

/* ── Water quality badge ─────── */
.wq-badge {
  display: flex; flex-direction: column; align-items: center;
  padding: 20px 16px; gap: 8px;
}
.wq-ring {
  width: 80px; height: 80px; border-radius: 50%;
  border: 3px solid var(--green);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.8rem;
  box-shadow: 0 0 24px rgba(0,255,153,.3);
  transition: all .4s;
}
.wq-ring.unsafe { border-color: var(--danger); box-shadow: 0 0 24px rgba(255,61,61,.3); }
.wq-label {
  font-family: var(--mono); font-size: .72rem; letter-spacing: .15em;
  color: var(--green); text-transform: uppercase; transition: color .4s;
}
.wq-label.unsafe { color: var(--danger); }
.wq-sub { font-family: var(--mono); font-size: .62rem; color: var(--dim); }

/* ── Detail rows ─────────────── */
.detail-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
  font-family: var(--mono); font-size: .68rem;
}
.detail-row:last-child { border-bottom: none; }
.detail-key { color: var(--dim); }
.detail-val { color: var(--txt); }
.detail-val.accent { color: var(--accent); }

/* ── History chart ───────────── */
#history-canvas {
  width: 100%; height: 80px;
  display: block;
}

/* ── Log ─────────────────────── */
#event-log {
  list-style: none;
  font-family: var(--mono); font-size: .65rem;
  max-height: 160px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 3px;
}
#event-log::-webkit-scrollbar { width: 3px; }
#event-log::-webkit-scrollbar-thumb { background: var(--border); }
#event-log li {
  padding: 3px 8px;
  border-left: 2px solid var(--border); color: var(--dim);
  transition: border-color .3s, color .3s;
}
#event-log li.fresh { border-left-color: var(--green); color: var(--txt); }
#event-log li.alert { border-left-color: var(--warn); color: var(--warn); }

/* ── Threshold slider ─────────── */
.slider-row { display: flex; align-items: center; gap: 10px; margin-top: 6px; }
input[type=range] {
  flex: 1; -webkit-appearance: none;
  height: 3px; background: var(--border); border-radius: 2px; outline: none; cursor: pointer;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 13px; height: 13px; border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 8px rgba(0,229,255,.5);
  cursor: pointer; transition: transform .15s;
}
input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.4); }
.slider-val { font-family: var(--mono); font-size: .68rem; color: var(--accent); min-width: 34px; text-align: right; }

/* ── Buttons ─────────────────── */
.btn-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px; }
.btn {
  font-family: var(--mono); font-size: .68rem; letter-spacing: .05em;
  padding: 6px 14px; border-radius: var(--r);
  border: 1px solid var(--border); background: transparent;
  color: var(--txt); cursor: pointer; transition: all .18s;
}
.btn:hover            { border-color: var(--accent); color: var(--accent); box-shadow: var(--glow); }
.btn.primary          { border-color: var(--accent); color: var(--accent); }
.btn.primary:hover    { background: rgba(0,229,255,.08); }
.btn.danger           { border-color: var(--danger); color: var(--danger); }
.btn.danger:hover     { background: rgba(255,61,61,.08); }

/* ── Image upload ────────────── */
.drop-zone {
  border: 2px dashed var(--border);
  border-radius: var(--r);
  padding: 16px 10px;
  text-align: center;
  cursor: pointer;
  transition: all .2s;
  position: relative;
}
.drop-zone:hover, .drop-zone.drag-over {
  border-color: var(--accent);
  background: rgba(0,229,255,.04);
  box-shadow: var(--glow);
}
.drop-zone input[type=file] {
  position: absolute; inset: 0;
  opacity: 0; cursor: pointer; width: 100%; height: 100%;
}
.drop-icon  { font-size: 1.5rem; margin-bottom: 4px; }
.drop-text  { font-family: var(--mono); font-size: .65rem; color: var(--dim); }
.drop-text b{ color: var(--accent); }

#upload-preview {
  width: 100%; border-radius: var(--r);
  border: 1px solid var(--border);
  margin-top: 10px; display: none;
}
.upload-result {
  margin-top: 10px;
  font-family: var(--mono); font-size: .68rem;
  color: var(--dim); display: none;
}
.upload-result.done { display: block; color: var(--green); }
.upload-result.err  { display: block; color: var(--danger); }
.upload-meta {
  display: flex; justify-content: space-between;
  font-family: var(--mono); font-size: .65rem;
  margin-top: 6px; color: var(--dim);
}
.upload-meta b { color: var(--accent); }

/* ── Footer ──────────────────── */
footer {
  position: relative; z-index: 1;
  text-align: center; padding: 14px;
  font-family: var(--mono); font-size: .62rem; color: var(--dim);
  border-top: 1px solid var(--border);
}
footer span { color: var(--accent); }
</style>
</head>
<body>

<!-- ── HEADER ──────────────────────────────────────── -->
<header>
  <div class="brand">
    <div class="brand-icon">&#128300;</div>
    <div>
      <div class="brand-name">Microplastic <span>Detection</span></div>
      <div class="brand-sub">Lensless Digital Holography · OV2640 · YOLOv8</div>
    </div>
  </div>
  <div class="header-right">
    <div class="pill" id="conn-pill"><div class="dot"></div><span id="conn-txt">CONNECTING</span></div>
    <div class="pill" style="border-color:var(--accent);color:var(--accent);">
      ESP32 &nbsp;<span style="color:var(--txt);">{{ esp32_ip }}</span>
    </div>
  </div>
</header>

<!-- ── MAIN ────────────────────────────────────────── -->
<main>

  <!-- LEFT: video feed -->
  <div style="display:flex;flex-direction:column;gap:14px;">
    <div class="card">
      <div class="card-head">
        <span class="card-title">&#9654; Live Detection Feed</span>
        <span class="pill online"><div class="dot"></div>STREAMING</span>
      </div>
      <div class="video-wrap">
        <div class="vid-overlay vid-tl">&#9679; REC &nbsp; MICROPLASTIC</div>
        <div class="vid-overlay vid-tr" id="fps-overlay">— fps</div>
        <div class="vid-overlay vid-bl" id="res-overlay">OV2640 · SVGA</div>
        <div class="vid-overlay vid-br" id="inf-overlay">inf —ms</div>
        <img id="feed" src="/video_feed" alt="live feed">
      </div>
      <div class="stat-row">
        <div class="stat">
          <div class="stat-val green" id="frame-count">0</div>
          <div class="stat-lbl">FRAMES</div>
        </div>
        <div class="stat">
          <div class="stat-val" id="fps-stat">0.0</div>
          <div class="stat-lbl">FPS</div>
        </div>
        <div class="stat">
          <div class="stat-val warn" id="runtime-stat">00:00</div>
          <div class="stat-lbl">RUNTIME</div>
        </div>
      </div>
    </div>

    <!-- Detection history chart -->
    <div class="card">
      <div class="card-head">
        <span class="card-title">&#9632; Detection History</span>
        <span class="pill" id="total-badge">Total: 0</span>
      </div>
      <div class="card-body" style="padding:12px 16px 14px;">
        <canvas id="history-canvas"></canvas>
      </div>
    </div>
  </div>

  <!-- RIGHT: controls + stats -->
  <div class="right">

    <!-- Water quality -->
    <div class="card">
      <div class="card-head"><span class="card-title">&#9675; Water Quality</span></div>
      <div class="card-body" style="padding:0;">
        <div class="wq-badge">
          <div class="wq-ring" id="wq-ring">&#128167;</div>
          <div class="wq-label" id="wq-label">SAFE</div>
          <div class="wq-sub" id="wq-sub">0 particles detected</div>
        </div>
      </div>
    </div>

    <!-- Per-frame detection -->
    <div class="card">
      <div class="card-head"><span class="card-title">&#128202; Current Frame</span></div>
      <div class="card-body">
        <div class="stat-row" style="padding:0;border:none;grid-template-columns:1fr 1fr;gap:12px;">
          <div class="stat">
            <div class="stat-val green" id="frame-detect">0</div>
            <div class="stat-lbl">THIS FRAME</div>
          </div>
          <div class="stat">
            <div class="stat-val" id="total-detect">0</div>
            <div class="stat-lbl">TOTAL</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Model info -->
    <div class="card">
      <div class="card-head"><span class="card-title">&#9670; Model Config</span></div>
      <div class="card-body">
        <div class="detail-row"><span class="detail-key">Model</span><span class="detail-val accent">{{ model_name }}</span></div>
        <div class="detail-row"><span class="detail-key">Confidence</span><span class="detail-val accent" id="conf-display">{{ confidence }}</span></div>
        <div class="detail-row"><span class="detail-key">Stream</span><span class="detail-val">:81/stream</span></div>
        <div class="detail-row"><span class="detail-key">Dashboard</span><span class="detail-val">:{{ port }}</span></div>
        <div class="detail-row"><span class="detail-key">Threshold</span><span class="detail-val accent">&gt;10 particles (last 10 frames)</span></div>

        <div style="margin-top:12px;">
          <div style="font-family:var(--mono);font-size:.62rem;color:var(--dim);margin-bottom:6px;">CONF THRESHOLD</div>
          <div class="slider-row">
            <input type="range" id="conf-slider" min="5" max="95" value="{{ conf_int }}" oninput="updateConf(this.value)">
            <span class="slider-val" id="conf-val">{{ confidence }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Controls -->
    <div class="card">
      <div class="card-head"><span class="card-title">&#9881; Controls</span></div>
      <div class="card-body">
        <div class="btn-row">
          <button class="btn primary" onclick="resetCount()">&#8635; Reset Count</button>
          <button class="btn" onclick="saveSnapshot()">&#9632; Snapshot</button>
          <button class="btn danger" onclick="clearLog()">CLR Log</button>
        </div>
      </div>
    </div>

    <!-- Image upload & inference -->
    <div class="card">
      <div class="card-head">
        <span class="card-title">&#128247; Image Inference</span>
        <span class="pill" id="upload-status-pill">IDLE</span>
      </div>
      <div class="card-body">
        <div class="drop-zone" id="drop-zone">
          <input type="file" id="img-upload" accept="image/*" onchange="handleUpload(this.files[0])">
          <div class="drop-icon">&#128247;</div>
          <div class="drop-text">Drop image or <b>click to browse</b><br>.jpg .png .bmp supported</div>
        </div>
        <img id="upload-preview" src="" alt="preview">
        <div class="upload-result" id="upload-result"></div>
        <div class="upload-meta" id="upload-meta" style="display:none;">
          <span>Particles: <b id="um-count">—</b></span>
          <span>Conf avg: <b id="um-conf">—</b></span>
          <span>Inf: <b id="um-inf">—</b>ms</span>
        </div>
      </div>
    </div>

    <!-- Event log -->
    <div class="card">
      <div class="card-head">
        <span class="card-title">&#9632; Event Log</span>
        <button class="btn" style="font-size:.6rem;padding:3px 8px;" onclick="clearLog()">CLR</button>
      </div>
      <div class="card-body" style="padding:10px 12px;">
        <ul id="event-log"></ul>
      </div>
    </div>

  </div><!-- /right -->
</main>

<!-- ── FOOTER ───────────────────────────────────────── -->
<footer>
  <span>MicroPlastic Detection System</span> · ESP32-CAM OV2640 · YOLOv8-nano · SDG 6.3 · Detection range 10μm+ · HMPD dataset 15,106 patches
</footer>

<script>
var confValue = {{ conf_float }};
var historyData = [];
var maxHistory  = 60;

// ── chart canvas ──────────────────────────────────────
var canvas  = document.getElementById('history-canvas');
var ctx     = canvas.getContext('2d');

function drawChart() {
  canvas.width  = canvas.offsetWidth;
  canvas.height = 80;
  var W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);

  if (historyData.length < 2) return;
  var max = Math.max.apply(null, historyData) || 1;

  // grid
  ctx.strokeStyle = 'rgba(30,45,60,.8)';
  ctx.lineWidth = 1;
  for (var g = 0; g <= 4; g++) {
    var y = H - (g/4)*H;
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke();
  }

  // fill
  var grad = ctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0,'rgba(0,229,255,.25)');
  grad.addColorStop(1,'rgba(0,229,255,0)');
  ctx.beginPath();
  var step = W / (maxHistory - 1);
  ctx.moveTo(0, H);
  for (var i = 0; i < historyData.length; i++) {
    var x = i * step;
    var y = H - (historyData[i]/max)*H*.9;
    if (i === 0) ctx.lineTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.lineTo((historyData.length-1)*step, H);
  ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();

  // line
  ctx.beginPath();
  ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 2;
  for (var i = 0; i < historyData.length; i++) {
    var x = i * step;
    var y = H - (historyData[i]/max)*H*.9;
    if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();
}

// ── poll stats every 1s ───────────────────────────────
function pollStats() {
  fetch('/stats')
    .then(function(r){ return r.json(); })
    .then(function(d) {
      document.getElementById('fps-stat').textContent     = d.fps.toFixed(1);
      document.getElementById('fps-overlay').textContent  = d.fps.toFixed(1) + ' fps';
      document.getElementById('runtime-stat').textContent = d.runtime;
      document.getElementById('frame-count').textContent  = d.frames;
      document.getElementById('frame-detect').textContent = d.frame_detections;
      document.getElementById('total-detect').textContent = d.total;
      document.getElementById('total-badge').textContent  = 'Total: ' + d.total;
      document.getElementById('inf-overlay').textContent  = 'inf ' + d.inference_ms + 'ms';

      // water quality based on backend 10-frame particle total calculation
      var unsafe = d.status === 'UNSAFE';
      var ring  = document.getElementById('wq-ring');
      var label = document.getElementById('wq-label');
      var sub   = document.getElementById('wq-sub');
      ring.className  = 'wq-ring'  + (unsafe ? ' unsafe' : '');
      label.className = 'wq-label' + (unsafe ? ' unsafe' : '');
      label.textContent = d.status;
      sub.textContent   = 'Win' + d.window_size + ': ' + d.window_total + ' particles | frame: ' + d.frame_detections;

      // connection pill
      var pill = document.getElementById('conn-pill');
      var txt  = document.getElementById('conn-txt');
      if (d.connected) {
        pill.className = 'pill online'; txt.textContent = 'ONLINE';
      } else {
        pill.className = 'pill offline'; txt.textContent = 'OFFLINE';
      }

      // history
      historyData.push(d.frame_detections);
      if (historyData.length > maxHistory) historyData.shift();
      drawChart();

      // alert log on spike
      if (d.frame_detections > 5) {
        addLog('SPIKE: ' + d.frame_detections + ' particles in frame', true);
      }
    })
    .catch(function(){ });
}

setInterval(pollStats, 1000);
pollStats();

// ── event log ─────────────────────────────────────────
function addLog(msg, alert) {
  var li   = document.createElement('li');
  var ts   = new Date().toLocaleTimeString();
  li.textContent = '[' + ts + '] ' + msg;
  li.className   = alert ? 'alert' : 'fresh';
  var log = document.getElementById('event-log');
  log.insertBefore(li, log.firstChild);
  while (log.children.length > 40) log.removeChild(log.lastChild);
  setTimeout(function(){ if (li.className === 'fresh') li.className = ''; }, 3000);
}

// ── confidence slider ─────────────────────────────────
function updateConf(val) {
  var f = (val / 100).toFixed(2);
  document.getElementById('conf-val').textContent     = f;
  document.getElementById('conf-display').textContent = f;
  fetch('/set_conf?v=' + f)
    .then(function(){ addLog('Confidence threshold → ' + f, false); })
    .catch(function(){});
}

// ── reset count ───────────────────────────────────────
function resetCount() {
  fetch('/reset')
    .then(function(){ addLog('Detection count reset to 0', false); })
    .catch(function(){});
}

// ── snapshot ──────────────────────────────────────────
function saveSnapshot() {
  window.open('/snapshot', '_blank');
  addLog('Snapshot saved', false);
}

// ── clear log ─────────────────────────────────────────
function clearLog() {
  document.getElementById('event-log').innerHTML = '';
}

// ── image upload & inference ──────────────────────────
var dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('dragover',  function(e){ e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', function(){  dropZone.classList.remove('drag-over'); });
dropZone.addEventListener('drop', function(e) {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  var file = e.dataTransfer.files[0];
  if (file) handleUpload(file);
});

function handleUpload(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showUploadResult('Not an image file', true);
    return;
  }

  // show local preview immediately
  var reader = new FileReader();
  reader.onload = function(e) {
    var prev = document.getElementById('upload-preview');
    prev.src = e.target.result;
    prev.style.display = 'block';
  };
  reader.readAsDataURL(file);

  // update pill
  var pill = document.getElementById('upload-status-pill');
  pill.textContent = 'PROCESSING';
  pill.style.borderColor = 'var(--warn)';
  pill.style.color       = 'var(--warn)';

  addLog('Image upload: ' + file.name, false);

  var fd = new FormData();
  fd.append('image', file);

  fetch('/infer_image', { method: 'POST', body: fd })
    .then(function(r) {
      if (!r.ok) throw new Error('Server error ' + r.status);
      return r.json();
    })
    .then(function(data) {
      // swap preview for annotated result
      var prev = document.getElementById('upload-preview');
      prev.src = 'data:image/jpeg;base64,' + data.image_b64;
      prev.style.display = 'block';

      document.getElementById('um-count').textContent = data.count;
      document.getElementById('um-conf').textContent  = data.avg_conf > 0 ? data.avg_conf.toFixed(2) : '—';
      document.getElementById('um-inf').textContent   = data.inference_ms;
      document.getElementById('upload-meta').style.display = 'flex';

      pill.textContent       = 'DONE';
      pill.style.borderColor = 'var(--green)';
      pill.style.color       = 'var(--green)';

      showUploadResult('Found ' + data.count + ' particle(s)  ·  ' + data.inference_ms + 'ms', false);
      addLog('Inference done: ' + data.count + ' particles  avg conf ' + (data.avg_conf > 0 ? data.avg_conf.toFixed(2) : 'n/a'), false);
    })
    .catch(function(err) {
      pill.textContent       = 'ERROR';
      pill.style.borderColor = 'var(--danger)';
      pill.style.color       = 'var(--danger)';
      showUploadResult('Upload failed: ' + err, true);
      addLog('Image inference error: ' + err, true);
    });
}

function showUploadResult(msg, isErr) {
  var el = document.getElementById('upload-result');
  el.textContent = msg;
  el.className   = 'upload-result ' + (isErr ? 'err' : 'done');
}

// ── resize chart on window resize ────────────────────
window.addEventListener('resize', drawChart);

addLog('Microplastic Detection dashboard initialised', false);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Detection Application
# ---------------------------------------------------------------------------
class MicroplasticDetectionApp:
    """Real-time microplastic detection via ESP32-CAM + YOLOv8."""

    def __init__(self, esp32_ip, model_path, confidence, port=5000):
        self.esp32_ip     = esp32_ip
        self.esp32_stream = f"http://{esp32_ip}:81/stream"
        self.model_path   = Path(model_path)
        self.confidence   = confidence
        self.port         = port

        self.detection_total    = 0
        self.frame_detections   = 0   # detections in the most recent frame
        self.frame_count        = 0
        self.inference_ms       = 0
        self.window_size        = 10
        self.unsafe_threshold   = 10
        self.count_buffer       = collections.deque(maxlen=self.window_size)
        self.rolling_avg        = 0.0
        self.window_total       = 0
        self.status             = "SAFE"
        self.start_time         = time.time()
        self.fps                = 0.0
        self.connected          = False
        self._lock              = threading.Lock()

        self.model = self._load_model()
        self.app   = Flask(__name__)
        self._setup_routes()

    # ---- model ----
    def _load_model(self):
        print(f"\nLoading model: {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        model = YOLO(str(self.model_path))
        print(f"Model loaded OK")
        return model

    # ---- routes ----
    def _setup_routes(self):

        @self.app.route('/')
        def index():
            return render_template_string(
                HTML_TEMPLATE,
                esp32_ip   = self.esp32_ip,
                model_name = self.model_path.name,
                confidence = f"{self.confidence:.2f}",
                conf_int   = int(self.confidence * 100),
                conf_float = self.confidence,
                port       = self.port,
            )

        @self.app.route('/video_feed')
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.route('/stats')
        def stats():
            runtime = time.time() - self.start_time
            mins, secs = divmod(int(runtime), 60)
            with self._lock:
                return jsonify(
                    total           = self.detection_total,
                    frame_detections= self.frame_detections,
                    fps             = round(self.fps, 1),
                    frames          = self.frame_count,
                    inference_ms    = self.inference_ms,
                  rolling_avg     = self.rolling_avg,
                  window_total    = self.window_total,
                  window_size     = self.window_size,
                  unsafe_threshold= self.unsafe_threshold,
                  status          = self.status,
                    runtime         = f"{mins:02d}:{secs:02d}",
                    connected       = self.connected,
                )

        @self.app.route('/reset')
        def reset():
            with self._lock:
                self.detection_total  = 0
                self.frame_detections = 0
            self.count_buffer.clear()
            self.rolling_avg = 0.0
            self.window_total = 0
            self.status = "SAFE"
            return ('', 204)

        @self.app.route('/set_conf')
        def set_conf():
            from flask import request
            try:
                v = float(request.args.get('v', self.confidence))
                self.confidence = max(0.05, min(0.95, v))
            except ValueError:
                pass
            return ('', 204)

        @self.app.route('/snapshot')
        def snapshot():
            frame = getattr(self, '_latest_frame', None)
            if frame is None:
                return ('No frame available', 503)
            ret, buf = cv2.imencode('.jpg', frame)
            if not ret:
                return ('Encode failed', 500)
            return send_file(
                io.BytesIO(buf.tobytes()),
                mimetype='image/jpeg',
                as_attachment=True,
                download_name=f"Microplastic Detection_{int(time.time())}.jpg"
            )

        @self.app.route('/infer_image', methods=['POST'])
        def infer_image():
            """Run YOLOv8 inference on an uploaded image, return annotated JPEG + stats."""
            if 'image' not in request.files:
                return jsonify(error='No image field'), 400

            file  = request.files['image']
            data  = file.read()
            arr   = cv2.imdecode(
                        __import__('numpy').frombuffer(data, __import__('numpy').uint8),
                        cv2.IMREAD_COLOR
                    )
            if arr is None:
                return jsonify(error='Could not decode image'), 400

            t0      = time.time()
            results = self.model(arr, conf=self.confidence, verbose=False)
            inf_ms  = int((time.time() - t0) * 1000)

            boxes    = results[0].boxes
            count    = len(boxes)
            avg_conf = float(boxes.conf.mean()) if count > 0 else 0.0

            annotated = results[0].plot()
            ret, buf  = cv2.imencode('.jpg', annotated)
            if not ret:
                return jsonify(error='Encode failed'), 500

            img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

            with self._lock:
              # Uploaded image inference should not change live stream status metrics.
              pass

            return jsonify(
                count        = count,
                avg_conf     = round(avg_conf, 4),
                inference_ms = inf_ms,
                image_b64    = img_b64,
            )

    # ---- frame generator ----
    def _generate_frames(self):
        cap = cv2.VideoCapture(self.esp32_stream)

        if not cap.isOpened():
            print(f"Cannot connect to stream: {self.esp32_stream}")
            self.connected = False
            return

        self.connected = True
        print(f"Stream connected: {self.esp32_stream}")

        frame_times = collections.deque(maxlen=30)

        while True:
            ok, frame = cap.read()
            if not ok:
                self.connected = False
                time.sleep(0.1)
                # attempt reconnect
                cap.release()
                cap = cv2.VideoCapture(self.esp32_stream)
                if cap.isOpened():
                    self.connected = True
                continue

            self.connected = True
            t0      = time.time()
            results = self.model(frame, conf=self.confidence, verbose=False)
            t1      = time.time()

            boxes = results[0].boxes
            n     = len(boxes)

            with self._lock:
                self.frame_detections = n
                self.detection_total += n
                self.frame_count     += 1
                self.inference_ms     = int((t1 - t0) * 1000)
                self.count_buffer.append(n)
                self.window_total = sum(self.count_buffer)
                self.rolling_avg = self.window_total / len(self.count_buffer)
                # Only evaluate UNSAFE when a full 10-frame window is available.
                if len(self.count_buffer) == self.window_size and self.window_total > self.unsafe_threshold:
                    self.status = "UNSAFE"
                else:
                    self.status = "SAFE"

            frame_times.append(time.time())
            if len(frame_times) > 1:
                self.fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])

            annotated = results[0].plot()

            # HUD overlays
            cv2.putText(annotated, f"Particles: {n}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 229, 255), 2)
            cv2.putText(annotated, f"Total: {self.detection_total}",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 153), 2)
            cv2.putText(annotated, f"FPS: {self.fps:.1f}  Inf: {self.inference_ms}ms",
                        (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            self._latest_frame = annotated.copy()

            ret, buf = cv2.imencode('.jpg', annotated)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

        cap.release()

    # ---- run ----
    def run(self):
        print("\n" + "=" * 60)
        print("  Microplastic Detection System")
        print("=" * 60)
        print(f"  ESP32 stream  →  {self.esp32_stream}")
        print(f"  Model         →  {self.model_path.name}")
        print(f"  Confidence    →  {self.confidence}")
        print(f"  Dashboard     →  http://localhost:{self.port}")
        print("=" * 60)
        print("  Ctrl+C to stop\n")
        self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Microplastic Detection System')
    parser.add_argument('--esp32', type=str,   default='192.168.29.9',
                        help='ESP32-CAM IP (default: 192.168.29.9)')
    parser.add_argument('--model', type=str,   default='yolov8_microplastic_trained.pt',
                        help='Path to trained YOLOv8 model')
    parser.add_argument('--conf',  type=float, default=0.25,
                        help='Confidence threshold 0.25–1 (default: 0.25)')
    parser.add_argument('--port',  type=int,   default=5000,
                        help='Dashboard port (default: 5000)')
    args = parser.parse_args()

    try:
        app = MicroplasticDetectionApp(
            esp32_ip   = args.esp32,
            model_path = args.model,
            confidence = args.conf,
            port       = args.port,
        )
        app.run()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\nError: {e}")
        import sys
        sys.exit(1)