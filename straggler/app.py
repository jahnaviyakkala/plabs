"""
dashboard/app.py
----------------
Lightweight Flask dashboard for PLABS straggler monitoring.

Routes:
  GET /            → main dashboard (HTML)
  GET /api/metrics → JSON of latest model metrics
  GET /api/events  → JSON of detection events from last Algorithm 1 run
  GET /api/stream  → Server-sent events for live straggler feed (optional)

Usage:
  python dashboard/app.py

Open http://localhost:5050 in browser.
"""

import os, sys, json
import pandas as pd
from flask import Flask, jsonify, render_template_string, Response
import queue, threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

app = Flask(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

# ── SSE event queue (populated by detector.py poll_cb) ────────────────────────
_event_queue: queue.Queue = queue.Queue(maxsize=200)


def push_event(payload: dict) -> None:
    """Call this from detector.py poll_cb to stream events to dashboard."""
    try:
        _event_queue.put_nowait(payload)
    except queue.Full:
        _event_queue.get_nowait()
        _event_queue.put_nowait(payload)


# ── HTML template ─────────────────────────────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="10">
<title>PLABS – Straggler Dashboard</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',Arial,sans-serif;background:#0d1117;color:#e6edf3}
  header{background:#161b22;border-bottom:1px solid #30363d;padding:14px 28px;
         display:flex;align-items:center;gap:12px}
  header h1{font-size:1.3rem;font-weight:600;color:#58a6ff}
  header span{font-size:.85rem;color:#8b949e}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
        gap:16px;padding:24px}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:18px}
  .card h3{font-size:.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;
           margin-bottom:8px}
  .card .val{font-size:2rem;font-weight:700;color:#58a6ff}
  .card .sub{font-size:.8rem;color:#8b949e;margin-top:4px}
  .section{padding:0 24px 24px}
  .section h2{font-size:1rem;color:#c9d1d9;margin-bottom:12px;
              border-bottom:1px solid #30363d;padding-bottom:6px}
  table{width:100%;border-collapse:collapse;font-size:.875rem}
  th{background:#21262d;color:#8b949e;padding:8px 12px;text-align:left;
     font-weight:500;font-size:.8rem;text-transform:uppercase}
  td{padding:8px 12px;border-bottom:1px solid #21262d}
  tr:hover td{background:#161b22}
  .badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.75rem;
         font-weight:600}
  .badge-good{background:#1a472a;color:#56d364}
  .badge-warn{background:#3a2a00;color:#e3b341}
  .badge-bad {background:#4a1515;color:#f85149}
  img{max-width:100%;border-radius:6px;border:1px solid #30363d;margin-top:8px}
  .plot-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media(max-width:700px){.plot-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>⚡ PLABS – Real-time Straggler Detection</h1>
  <span>AUtool+ Framework · Hadoop MapReduce</span>
</header>

<div class="grid" id="kpi-grid">
  <div class="card">
    <h3>Models Trained</h3>
    <div class="val" id="n-models">–</div>
    <div class="sub">ML + DL</div>
  </div>
  <div class="card">
    <h3>Best F1-Score</h3>
    <div class="val" id="best-f1">–</div>
    <div class="sub" id="best-model">–</div>
  </div>
  <div class="card">
    <h3>Detection Events</h3>
    <div class="val" id="n-events">–</div>
    <div class="sub">across all polling rounds</div>
  </div>
  <div class="card">
    <h3>Peak Stragglers</h3>
    <div class="val" id="peak-strag">–</div>
    <div class="sub">in a single poll</div>
  </div>
</div>

<div class="section">
  <h2>Model Performance</h2>
  <table id="metrics-table">
    <thead><tr>
      <th>Model</th><th>Accuracy</th><th>Precision</th>
      <th>Recall</th><th>F1</th>
    </tr></thead>
    <tbody id="metrics-body"><tr><td colspan="5">Loading…</td></tr></tbody>
  </table>
</div>

<div class="section">
  <h2>Detection Timeline – Algorithm 1 Replay</h2>
  <table id="events-table">
    <thead><tr>
      <th>Poll Round</th><th>Job Progress (%)</th>
      <th>Tasks Seen</th><th>New Stragglers</th>
      <th>Cumulative Stragglers</th>
    </tr></thead>
    <tbody id="events-body"><tr><td colspan="5">Loading…</td></tr></tbody>
  </table>
</div>

<div class="section">
  <h2>Visualisations</h2>
  <div class="plot-grid">
    <div><img src="/plot/metric_bars.png" alt="Metric bars" onerror="this.style.display='none'"></div>
    <div><img src="/plot/roc_curves.png" alt="ROC curves" onerror="this.style.display='none'"></div>
    <div><img src="/plot/feat_imp_RandomForest.png" alt="Feature importance" onerror="this.style.display='none'"></div>
    <div><img src="/plot/detection_timeline.png" alt="Detection timeline" onerror="this.style.display='none'"></div>
  </div>
</div>

<script>
async function loadMetrics(){
  const r=await fetch('/api/metrics'); const d=await r.json();
  if(!d.length){return;}
  document.getElementById('n-models').textContent=d.length;
  const best=d.reduce((a,b)=>((b.f1||0)>(a.f1||0)?b:a));
  document.getElementById('best-f1').textContent=(best.f1*100).toFixed(1)+'%';
  document.getElementById('best-model').textContent=best.model||'';
  const tbody=document.getElementById('metrics-body');
  tbody.innerHTML=d.map(r=>`<tr>
    <td>${r.model}</td>
    <td>${(r.accuracy*100).toFixed(2)}%</td>
    <td>${(r.precision*100).toFixed(2)}%</td>
    <td>${(r.recall*100).toFixed(2)}%</td>
    <td><span class="badge ${r.f1>0.85?'badge-good':r.f1>0.6?'badge-warn':'badge-bad'}">${(r.f1*100).toFixed(2)}%</span></td>
  </tr>`).join('');
}
async function loadEvents(){
  const r=await fetch('/api/events'); const d=await r.json();
  if(!d.length){return;}
  document.getElementById('n-events').textContent=d.reduce((s,x)=>s+(x.new_stragglers||0),0);
  document.getElementById('peak-strag').textContent=Math.max(...d.map(x=>x.new_stragglers||0));
  const tbody=document.getElementById('events-body');
  tbody.innerHTML=d.slice(-30).map(r=>`<tr>
    <td>${r.poll_round}</td>
    <td>${(r.job_progress||0).toFixed(1)}</td>
    <td>${r.tasks_seen||0}</td>
    <td><b style="color:${r.new_stragglers>0?'#f85149':'#56d364'}">${r.new_stragglers||0}</b></td>
    <td>${r.cumulative_stragglers||0}</td>
  </tr>`).join('');
}
loadMetrics(); loadEvents();
</script>
</body>
</html>
"""


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/metrics")
def api_metrics():
    path = os.path.join(OUTPUT_DIR, "metrics.csv")
    if not os.path.exists(path):
        return jsonify([])
    df = pd.read_csv(path).reset_index()
    if "model" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "model"})
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/events")
def api_events():
    path = os.path.join(OUTPUT_DIR, "detection_events.csv")
    if not os.path.exists(path):
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.route("/plot/<filename>")
def serve_plot(filename):
    from flask import send_from_directory
    return send_from_directory(os.path.join(OUTPUT_DIR, "plots"), filename)


@app.route("/api/stream")
def stream():
    """Server-sent events for live straggler feed."""
    def generate():
        while True:
            try:
                payload = _event_queue.get(timeout=30)
                yield f"data: {json.dumps(payload)}\n\n"
            except Exception:
                yield ": keepalive\n\n"
    return Response(generate(), mimetype="text/event-stream")


# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 5050))
    print(f"Dashboard running at  http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
