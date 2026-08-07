"""
Flask web UI for OpenROAD-flow-scripts.

Run from the flow/ directory:
    python3 ui/app.py

Then open http://localhost:5000 in a browser.
"""

import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import threading

import torch

from flask import Flask, Response, jsonify, render_template, request, send_file
from layout_parser import (
    find_lef_files, parse_lef_macros, get_available_stages,
    get_design_nickname, parse_def, parse_timing_reports,
)

app = Flask(__name__)

# Registry of all active subprocesses.  Populated by stream_command(); cleaned
# up when each process exits.  The signal handler iterates this to kill
# lingering Docker processes when Ctrl+C is pressed.
_active_procs: set = set()
_active_procs_lock = threading.Lock()


def _kill_active_procs():
    with _active_procs_lock:
        procs = list(_active_procs)
    for p in procs:
        try:
            # Kill the entire process group so that docker run (which docker_shell
            # execs into) and any grandchildren are all killed together, not just
            # the top-level bash shell that would otherwise leave them orphaned.
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            try:
                p.terminate()
            except Exception:
                pass
    for p in procs:
        try:
            p.wait(timeout=3)
        except Exception:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass


def _sigint_handler(sig, frame):
    _kill_active_procs()
    sys.exit(0)


signal.signal(signal.SIGINT, _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)

FLOW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# When the UI server runs inside the orfs-ui Docker container (run_ui.sh --docker),
# ORFS_HOST_FLOW_DIR is the flow/ path on the HOST.  docker_shell uses this to
# set the volume-mount source correctly (host path, not container /work path).
_HOST_FLOW_DIR = os.environ.get("ORFS_HOST_FLOW_DIR")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_CACHE: dict = {}


def _load_model(model_class, checkpoint: str, *init_args, **init_kwargs):
    """Load a model from checkpoint, caching it for the lifetime of the server."""
    if checkpoint in _MODEL_CACHE:
        return _MODEL_CACHE[checkpoint]
    model = model_class(*init_args, **init_kwargs)
    model.load_state_dict(torch.load(checkpoint, map_location=_DEVICE))
    model.to(_DEVICE)
    if _DEVICE.type == "cuda":
        model.half()  # FP16 inference on GPU
    model.eval()
    if shutil.which("g++") or shutil.which("c++"):
        try:
            model = torch.compile(model)
        except Exception:
            pass  # torch.compile is optional; skip if unsupported
    _MODEL_CACHE[checkpoint] = model
    return model


def stream_command(cmd):
    """Run a shell command and yield output lines as SSE messages."""
    env = os.environ.copy()
    if _HOST_FLOW_DIR:
        env["ORFS_HOST_FLOW_DIR"] = _HOST_FLOW_DIR
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=FLOW_DIR,
        env=env,
        start_new_session=True,  # own process group — killpg won't kill Flask
    )
    with _active_procs_lock:
        _active_procs.add(process)
    try:
        for line in iter(process.stdout.readline, ""):
            yield f"data: {line.rstrip()}\n\n"
        process.wait()
    finally:
        with _active_procs_lock:
            _active_procs.discard(process)
    yield f"data: \n\n"
    yield f"data: [EXIT {process.returncode}]\n\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/designs")
def list_designs():
    designs = []
    designs_root = os.path.join(FLOW_DIR, "designs")
    for platform_dir in sorted(glob.glob(os.path.join(designs_root, "*"))):
        platform = os.path.basename(platform_dir)
        if not os.path.isdir(platform_dir) or platform == "src":
            continue
        for design_dir in sorted(glob.glob(os.path.join(platform_dir, "*"))):
            design = os.path.basename(design_dir)
            if os.path.isfile(os.path.join(design_dir, "config.mk")):
                designs.append({"platform": platform, "design": design})
    return jsonify(designs)


@app.route("/api/run/<platform>/<design>")
def run_flow(platform, design):
    config = f"/work/designs/{platform}/{design}/config.mk"
    from_stage = request.args.get("from_stage", "")
    _stage_targets = {
        'floorplan': 'floorplan', 'place': 'place', 'cts': 'cts',
        'grt': 'grt', 'route': 'route', 'finish': 'finish',
    }
    target = _stage_targets.get(from_stage, "")
    cmd = f"util/docker_shell make DESIGN_CONFIG={config} DESIGN_HOME=/work/designs {target}".strip()
    return Response(stream_command(cmd), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.route("/api/timing/<platform>/<design>")
def timing(platform, design):
    data = parse_timing_reports(platform, design, FLOW_DIR)
    return jsonify(data)


@app.route("/api/compare")
def compare_designs():
    pa = request.args.get("platformA", "")
    da = request.args.get("designA", "")
    pb = request.args.get("platformB", "")
    db = request.args.get("designB", "")
    return jsonify({
        'a': parse_timing_reports(pa, da, FLOW_DIR) if pa and da else None,
        'b': parse_timing_reports(pb, db, FLOW_DIR) if pb and db else None,
    })


_DRC_CACHE: dict = {}

@app.route("/api/drc/violations/<platform>/<design>")
def drc_violations(platform, design):
    from layout_parser import get_design_nickname
    key = f"{platform}/{design}"
    if key in _DRC_CACHE:
        return jsonify(_DRC_CACHE[key])
    nickname = get_design_nickname(platform, design, FLOW_DIR)
    lyrdb = os.path.join(FLOW_DIR, "reports", platform, nickname, "base", "6_drc.lyrdb")
    if not os.path.exists(lyrdb):
        return jsonify({"violations": [], "error": "6_drc.lyrdb not found — run through finish first"})
    convert_script = os.path.join(FLOW_DIR, "util", "convertDrc.py")
    result = subprocess.run(
        ["python3", convert_script, lyrdb],
        capture_output=True, text=True, cwd=FLOW_DIR,
    )
    if result.returncode != 0:
        return jsonify({"violations": [], "error": result.stderr[:500]})
    import json as _json
    try:
        violations = _json.loads(result.stdout)
    except Exception:
        violations = []
    payload = {"violations": violations}
    _DRC_CACHE[key] = payload
    return jsonify(payload)


@app.route("/api/drc/<platform>/<design>")
def run_drc(platform, design):
    config = f"/work/designs/{platform}/{design}/config.mk"
    cmd = f"util/docker_shell make DESIGN_CONFIG={config} DESIGN_HOME=/work/designs drc"
    return Response(stream_command(cmd), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


def _run_congestion_inference(placement_abs, checkpoint_abs, out_npy_abs, out_img_abs):
    """Run U-Net inference in-process using the cached GPU model. Returns (lines, exit_code)."""
    import numpy as np
    sys.path.insert(0, os.path.join(FLOW_DIR, "ml", "congestion", "model"))
    from unet import CongestionUNet

    placement = np.load(placement_abs).astype(np.float32)
    model = _load_model(CongestionUNet, checkpoint_abs, in_channels=1, out_channels=10)

    x = torch.tensor(placement).unsqueeze(0).unsqueeze(0).to(_DEVICE)
    if _DEVICE.type == "cuda":
        x = x.half()
    with torch.no_grad():
        pred = model(x)
    congestion = pred.squeeze(0).float().cpu().numpy()

    np.save(out_npy_abs, congestion)

    layer_names = ["metal1","metal2","metal3","metal4","metal5",
                   "metal6","metal7","metal8","metal9","metal10"]
    lines = [f"Saved predicted congestion {congestion.shape}",
             f"Device: {_DEVICE}"]
    for i, name in enumerate(layer_names):
        lines.append(f"  {name:8s}: mean={congestion[i].mean()*100:.1f}%  max={congestion[i].max()*100:.1f}%")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(20, 4))
        gs = gridspec.GridSpec(2, 5, figure=fig)
        for i, name in enumerate(layer_names):
            ax = fig.add_subplot(gs[i // 5, i % 5])
            im = ax.imshow(congestion[i], vmin=0, vmax=1, cmap="hot", origin="lower")
            ax.set_title(name, fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.suptitle("Predicted Congestion per Routing Layer", fontsize=12)
        plt.tight_layout()
        plt.savefig(out_img_abs, dpi=150, bbox_inches="tight")
        plt.close()
        lines.append(f"Saved image -> {out_img_abs}")
    except ImportError:
        lines.append("matplotlib not available; skipping image")

    return lines, 0


@app.route("/api/congestion/predict/<platform>/<design>")
def predict_congestion(platform, design):
    placement_rel = f"ml/data/{platform}_{design}_placement.npy"
    checkpoint_abs = os.path.join(FLOW_DIR, "ml/congestion/model/checkpoints/best.pt")
    out_npy_abs = os.path.join(FLOW_DIR, f"ml/data/{platform}_{design}_predicted_congestion.npy")
    out_img_abs = os.path.join(FLOW_DIR, f"ml/data/{platform}_{design}_predicted_congestion.png")
    placement_abs = os.path.join(FLOW_DIR, placement_rel)

    def _infer():
        try:
            lines, code = _run_congestion_inference(
                placement_abs, checkpoint_abs, out_npy_abs, out_img_abs)
            for l in lines:
                yield f"data: {l}\n\n"
            yield f"data: \n\n"
            yield f"data: [EXIT {code}]\n\n"
        except Exception as e:
            yield f"data: ERROR: {e}\n\n"
            yield f"data: [EXIT 1]\n\n"

    def _extract_then_infer():
        nickname = get_design_nickname(platform, design, FLOW_DIR)
        odb = f"/work/results/{platform}/{nickname}/base/3_5_place_dp.odb"
        extract_cmd = (
            f"util/docker_shell openroad -python"
            f" /work/ml/congestion/data_collection/extract_placement.py"
            f" --odb {odb}"
            f" --out /work/ml/data/{platform}_{design}_placement.npy"
            f" --grid 64"
        )
        yield f"data: Extracting placement grid from {odb}...\n\n"
        proc = subprocess.Popen(
            extract_cmd, shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, cwd=FLOW_DIR,
        )
        for line in iter(proc.stdout.readline, ""):
            yield f"data: {line.rstrip()}\n\n"
        proc.wait()
        if proc.returncode != 0:
            yield f"data: ERROR: placement extraction failed\n\n"
            yield f"data: [EXIT 1]\n\n"
            return
        yield f"data: Extraction complete. Running prediction on {_DEVICE}...\n\n"
        yield from _infer()

    gen = _infer() if os.path.exists(placement_abs) else _extract_then_infer()
    return Response(gen, mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.route("/api/congestion/image/<platform>/<design>")
def congestion_image(platform, design):
    img_path = os.path.join(FLOW_DIR, f"ml/data/{platform}_{design}_predicted_congestion.png")
    if os.path.exists(img_path):
        return send_file(img_path, mimetype="image/png")
    return "Image not found — run Predict Congestion first.", 404


@app.route("/api/congestion/overlay/<platform>/<design>")
def congestion_overlay(platform, design):
    """Return a clean single-layer heatmap PNG suitable for canvas overlay.

    Query params:
      layer  0–9 (metal1–metal10), default 0
      cmap   matplotlib colormap name, default 'hot'
    """
    import numpy as np
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npy_path = os.path.join(FLOW_DIR, f"ml/data/{platform}_{design}_predicted_congestion.npy")
    if not os.path.exists(npy_path):
        return "Run Predict Congestion first.", 404

    layer = max(0, min(9, int(request.args.get("layer", "0"))))
    cmap  = request.args.get("cmap", "hot")

    data = np.load(npy_path)[layer]          # (64, 64) float32 in [0, 1]

    h, w = data.shape
    fig = plt.figure(figsize=(w / 64, h / 64), dpi=64)
    ax  = fig.add_axes([0, 0, 1, 1])        # fill entire figure
    ax.imshow(data, vmin=0, vmax=1, cmap=cmap, origin="lower", aspect="auto",
              interpolation="bilinear")
    ax.set_axis_off()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=64, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png", max_age=0)


@app.route("/api/floorplan/suggest/<platform>/<design>")
def suggest_floorplan(platform, design):
    graph = f"ml/data/{platform}_{design}_graph.npz"
    checkpoint = "ml/floorplan/model/checkpoints/best.pt"
    out_tcl = f"ml/data/{platform}_{design}_suggested.tcl"
    die_w = request.args.get("die_w", "2000")
    die_h = request.args.get("die_h", "2000")

    def _run_suggest():
        import numpy as np
        graph_abs = os.path.join(FLOW_DIR, graph)
        g = np.load(graph_abs, allow_pickle=True)
        num_macros = int(g["num_macros"][0])
        if num_macros == 0:
            yield f"data: This design has no hard macros (RAM, IP blocks, etc.).\n\n"
            yield f"data: Suggest Floorplan only applies to designs containing macro cells.\n\n"
            yield f"data: Designs with macros: ariane133, black_parrot, mempool_group, microwatt.\n\n"
            yield f"data: [EXIT 0]\n\n"
            return
        try:
            sys.path.insert(0, os.path.join(FLOW_DIR, "ml", "floorplan", "model"))
            from gnn import FloorplanGNN
            checkpoint_abs = os.path.join(FLOW_DIR, checkpoint)
            model = _load_model(FloorplanGNN, checkpoint_abs)

            node_feat = torch.tensor(g["node_features"], dtype=torch.float32).to(_DEVICE)
            edge_index = torch.tensor(g["edge_index"], dtype=torch.long).to(_DEVICE)
            node_names = g["node_names"]
            macro_mask = node_feat[:, 1] > 0.5
            batch = torch.zeros(node_feat.shape[0], dtype=torch.long).to(_DEVICE)
            if _DEVICE.type == "cuda":
                node_feat = node_feat.half()

            with torch.no_grad():
                coords = model(node_feat, edge_index, batch, macro_mask)
            coords_np = coords.float().cpu().numpy()
            macro_names = node_names[macro_mask.cpu().numpy()]

            die_w_f, die_h_f = float(die_w), float(die_h)
            tcl_lines = [
                "# Auto-generated macro placement suggestions",
                "# Review and edit before sourcing in your floorplan Tcl script",
                "",
            ]
            yield f"data: Suggested placement for {len(macro_names)} macros (die: {die_w_f}x{die_h_f} um) on {_DEVICE}:\n\n"
            for name, (x_norm, y_norm) in zip(macro_names, coords_np):
                x_um, y_um = x_norm * die_w_f, y_norm * die_h_f
                yield f"data:   {name}: x={x_um:.1f} um, y={y_um:.1f} um\n\n"
                tcl_lines.append(f"place_inst {name} {x_um:.3f} {y_um:.3f} R0")

            out_tcl_abs = os.path.join(FLOW_DIR, out_tcl)
            os.makedirs(os.path.dirname(out_tcl_abs), exist_ok=True)
            with open(out_tcl_abs, "w") as f:
                f.write("\n".join(tcl_lines) + "\n")
            yield f"data: Saved Tcl -> {out_tcl_abs}\n\n"
            yield f"data: [EXIT 0]\n\n"
        except Exception as e:
            yield f"data: ERROR: {e}\n\n"
            yield f"data: [EXIT 1]\n\n"

    def _extract_then_suggest():
        nickname = get_design_nickname(platform, design, FLOW_DIR)
        odb = f"/work/results/{platform}/{nickname}/base/1_synth.odb"
        extract_cmd = (
            f"util/docker_shell openroad -python"
            f" /work/ml/floorplan/data_collection/extract_netlist_graph.py"
            f" --odb {odb}"
            f" --out /work/ml/data/{platform}_{design}_graph.npz"
        )
        yield f"data: Extracting netlist graph from {odb}...\n\n"
        proc = subprocess.Popen(
            extract_cmd, shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, cwd=FLOW_DIR,
        )
        for line in iter(proc.stdout.readline, ""):
            yield f"data: {line.rstrip()}\n\n"
        proc.wait()
        if proc.returncode != 0:
            yield f"data: ERROR: graph extraction failed\n\n"
            yield f"data: [EXIT 1]\n\n"
            return
        yield f"data: Extraction complete.\n\n"
        yield from _run_suggest()

    graph_exists = os.path.exists(os.path.join(FLOW_DIR, graph))
    generator = _run_suggest() if graph_exists else _extract_then_suggest()
    return Response(generator, mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.route("/api/floorplan/overlay/<platform>/<design>")
def floorplan_overlay_data(platform, design):
    """Parse the suggested Tcl and return macro positions as JSON for canvas overlay."""
    tcl_path = os.path.join(FLOW_DIR, f"ml/data/{platform}_{design}_suggested.tcl")
    if not os.path.exists(tcl_path):
        return jsonify({'macros': []})
    macros = []
    with open(tcl_path) as f:
        for line in f:
            m = re.match(r'place_inst\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)', line.strip())
            if m:
                macros.append({
                    'name':   m.group(1),
                    'x_um':   float(m.group(2)),
                    'y_um':   float(m.group(3)),
                    'orient': m.group(4),
                })
    return jsonify({'macros': macros})


@app.route("/api/congestion/predict_positions", methods=["POST"])
def predict_congestion_positions():
    """Re-run U-Net with macro positions moved in the placement grid, return overlay PNG."""
    import numpy as np
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    body     = request.get_json(force=True)
    platform = body.get("platform", "")
    design   = body.get("design", "")
    die_w_um = float(body.get("die_w_um", 1))
    die_h_um = float(body.get("die_h_um", 1))
    layer    = max(0, min(9, int(body.get("layer", 0))))
    macros   = body.get("macros", [])

    placement_abs  = os.path.join(FLOW_DIR, f"ml/data/{platform}_{design}_placement.npy")
    checkpoint_abs = os.path.join(FLOW_DIR, "ml/congestion/model/checkpoints/best.pt")

    if not os.path.exists(placement_abs):
        return "Run Predict Congestion first.", 404
    if not os.path.exists(checkpoint_abs):
        return "No model checkpoint found.", 404

    grid = np.load(placement_abs).astype(np.float32).copy()  # (64, 64)
    G = grid.shape[0]

    for m in macros:
        w_c = max(1, int(m.get("w_um", 0) / die_w_um * G))
        h_c = max(1, int(m.get("h_um", 0) / die_h_um * G))
        # Remove from old position
        oc  = int(m["old_x_um"] / die_w_um * G)
        or_ = int(m["old_y_um"] / die_h_um * G)
        oc1, oc2 = max(0, oc), min(G, oc + w_c)
        or1, or2 = max(0, or_), min(G, or_ + h_c)
        saved = grid[or1:or2, oc1:oc2].copy()
        grid[or1:or2, oc1:oc2] = 0.0
        # Add at new position
        nc  = int(m["new_x_um"] / die_w_um * G)
        nr  = int(m["new_y_um"] / die_h_um * G)
        nc1, nc2 = max(0, nc), min(G, nc + w_c)
        nr1, nr2 = max(0, nr), min(G, nr + h_c)
        ch = min(saved.shape[0], nr2 - nr1)
        cw = min(saved.shape[1], nc2 - nc1)
        if ch > 0 and cw > 0:
            grid[nr1:nr1+ch, nc1:nc1+cw] = np.maximum(
                grid[nr1:nr1+ch, nc1:nc1+cw], saved[:ch, :cw])

    sys.path.insert(0, os.path.join(FLOW_DIR, "ml", "congestion", "model"))
    from unet import CongestionUNet
    model = _load_model(CongestionUNet, checkpoint_abs, in_channels=1, out_channels=10)

    x = torch.tensor(grid).unsqueeze(0).unsqueeze(0).to(_DEVICE)
    if _DEVICE.type == "cuda":
        x = x.half()
    with torch.no_grad():
        pred = model(x)
    data = pred.squeeze(0).float().cpu().numpy()[layer]  # (64, 64)

    fig = plt.figure(figsize=(1, 1), dpi=64)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.imshow(data, vmin=0, vmax=1, cmap="hot", origin="lower",
              aspect="auto", interpolation="bilinear")
    ax.set_axis_off()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=64, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png", max_age=0)


@app.route("/api/floorplan/export", methods=["POST"])
def floorplan_export():
    """Save edited macro positions as a Tcl place_inst script and return it."""
    body     = request.get_json(force=True)
    platform = body.get("platform", "")
    design   = body.get("design", "")
    macros   = body.get("macros", [])  # [{name, x_um, y_um, orient}]

    lines = [
        "# Edited macro placement — generated by OpenROAD Flow UI",
        "# Source this file at the start of your floorplan Tcl script",
        "",
    ]
    for m in macros:
        orient = m.get("orient", "R0")
        lines.append(f"place_inst {m['name']} {float(m['x_um']):.3f} {float(m['y_um']):.3f} {orient}")

    tcl_path = os.path.join(FLOW_DIR, f"ml/data/{platform}_{design}_suggested.tcl")
    os.makedirs(os.path.dirname(tcl_path), exist_ok=True)
    with open(tcl_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return "\n".join(lines) + "\n", 200, {"Content-Type": "text/plain"}


@app.route("/api/floorplan/tcl/<platform>/<design>")
def get_floorplan_tcl(platform, design):
    tcl_path = os.path.join(FLOW_DIR, f"ml/data/{platform}_{design}_suggested.tcl")
    if os.path.exists(tcl_path):
        with open(tcl_path) as f:
            return f.read(), 200, {"Content-Type": "text/plain"}
    return "Tcl file not found — run Suggest Floorplan first.", 404


@app.route("/api/autofix/timing/<platform>/<design>")
def autofix_timing(platform, design):
    """
    Autonomous timing closure loop.

    Reads WNS from the latest timing report; if timing is failing (WNS < 0),
    tries a schedule of make-variable adjustments and re-runs from grt after
    each one.  Streams progress as SSE and emits [EXIT 0] on success or
    [EXIT 1] after exhausting all attempts.

    Parameter schedule (each re-runs from grt):
      1. TNS_END_PERCENT=25
      2. TNS_END_PERCENT=50
      3. TNS_END_PERCENT=100
      4. CORE_UTILIZATION reduced by 5 % from config default (re-runs from place)
    """
    config_path = f"/work/designs/{platform}/{design}/config.mk"

    def _current_wns(last_make_target=None):
        data = parse_timing_reports(platform, design, FLOW_DIR)
        stages = data.get('stages', {})
        # After re-running GRT the 5_global_route.rpt is fresh but 6_finish.rpt is
        # stale (from the previous full flow).  Prioritise the freshest source.
        if last_make_target == 'grt':
            priority = ('Global Route', 'Final', 'CTS', 'Detailed Place')
        else:
            priority = ('Final', 'Global Route', 'CTS', 'Detailed Place')
        for label in priority:
            if label in stages and stages[label].get('wns') is not None:
                return stages[label]['wns']
        return None

    def _read_config_utilization():
        """Return CORE_UTILIZATION from config.mk, or the measured utilization
        from the timing reports as a fallback when config.mk doesn't set it."""
        cfg = os.path.join(FLOW_DIR, 'designs', platform, design, 'config.mk')
        if os.path.exists(cfg):
            with open(cfg) as f:
                for line in f:
                    m = re.match(r'^\s*CORE_UTILIZATION\s*[?:]?=\s*([\d.]+)', line)
                    if m:
                        return float(m.group(1))
        # config.mk doesn't set CORE_UTILIZATION — read the measured value from
        # the timing reports (the ODB design_area reports actual utilization %).
        data = parse_timing_reports(platform, design, FLOW_DIR)
        for label in ('Final', 'Global Route', 'CTS'):
            s = data.get('stages', {}).get(label, {})
            if s.get('utilization') is not None:
                return round(s['utilization'] * 100)
        return None

    def _run(extra_vars, clear_stage, make_target):
        vars_str = ' '.join(f'{k}={v}' for k, v in extra_vars.items())
        # Delete outputs of clear_stage (and downstream) so make re-runs from
        # that stage forward.  make_target controls where make stops.
        # We never use -B because that rebuilds every dependency including synth.
        nickname = get_design_nickname(platform, design, FLOW_DIR)
        results_dir = os.path.join(FLOW_DIR, 'results', platform, nickname, 'base')
        _stage_outputs = {
            'grt':   ['5_1_grt.odb', '5_1_grt.sdc'],
            'cts':   ['4_1_cts.odb', '4_cts.odb', '4_cts.sdc',
                      '5_1_grt.odb', '5_1_grt.sdc'],
            'place': ['3_place.odb', '3_5_place_dp.odb', '3_place.sdc',
                      '4_1_cts.odb', '4_cts.odb', '4_cts.sdc',
                      '5_1_grt.odb', '5_1_grt.sdc'],
        }
        for fname in _stage_outputs.get(clear_stage, []):
            fpath = os.path.join(results_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        yield f"data: Cleared {clear_stage} outputs — make will re-run from {clear_stage} through {make_target}.\n\n"
        cmd = (
            f"util/docker_shell make DESIGN_CONFIG={config_path}"
            f" DESIGN_HOME=/work/designs {vars_str} {make_target}"
        ).strip()
        yield f"data: Running: {cmd}\n\n"
        yield from stream_command(cmd)

    def _generate():
        wns = _current_wns()
        if wns is None:
            yield "data: No timing data found — run the flow at least through Global Route first.\n\n"
            yield "data: [EXIT 1]\n\n"
            return
        if wns >= 0:
            yield f"data: Timing already closed (WNS={wns:+.3f} ns). Nothing to do.\n\n"
            yield "data: [EXIT 0]\n\n"
            return

        yield f"data: === Autonomous Timing Closure ===\n\n"
        yield f"data: Starting WNS: {wns:+.3f} ns — attempting to close timing.\n\n"

        # Each tuple: (extra_vars, clear_stage, make_target, description)
        # clear_stage: ODB files to delete so make re-runs from there.
        # make_target: where make stops (may be further than clear_stage).
        # Running 'make grt' after clearing 'place' outputs causes make to
        # re-run place→cts→grt in one shot and refreshes 5_global_route.rpt.
        schedule = [
            ({'TNS_END_PERCENT': '25'},  'grt', 'grt', 'TNS_END_PERCENT=25 (re-run GRT)'),
            ({'TNS_END_PERCENT': '50'},  'grt', 'grt', 'TNS_END_PERCENT=50 (re-run GRT)'),
            ({'TNS_END_PERCENT': '100'}, 'grt', 'grt', 'TNS_END_PERCENT=100 (re-run GRT)'),
        ]

        # Add utilization-reduction attempt if we can read the current value.
        # Clearing from 'place' but running 'grt' causes make to chain
        # place→cts→grt and produce fresh timing in 5_global_route.rpt.
        base_util = _read_config_utilization()
        if base_util is not None:
            new_util = max(20, int(base_util) - 5)
            schedule.append(
                ({'TNS_END_PERCENT': '100', 'CORE_UTILIZATION': str(new_util)},
                 'place', 'grt',
                 f'CORE_UTILIZATION {int(base_util)}→{new_util}% (re-run place→cts→grt)')
            )

        last_make_target = None
        for attempt, (extra_vars, clear_stage, make_target, desc) in enumerate(schedule, 1):
            yield f"data: \n\n"
            yield f"data: --- Attempt {attempt}/{len(schedule)}: {desc} ---\n\n"

            exit_code = None
            for msg in _run(extra_vars, clear_stage, make_target):
                yield msg
                if '[EXIT ' in msg:
                    try:
                        exit_code = int(msg.split('[EXIT ')[1].split(']')[0])
                    except Exception:
                        exit_code = 1

            if exit_code != 0:
                yield f"data: Make exited with error — stopping.\n\n"
                yield f"data: [EXIT 1]\n\n"
                return

            last_make_target = make_target
            wns = _current_wns(last_make_target)
            if wns is None:
                yield "data: Could not read updated WNS after run.\n\n"
                continue

            source = 'GRT estimate' if last_make_target == 'grt' else 'post-route STA'
            yield f"data: \n\n"
            yield f"data: Updated WNS: {wns:+.3f} ns ({source})\n\n"

            if wns >= 0:
                yield f"data: Timing closed after {attempt} attempt(s)! WNS={wns:+.3f} ns\n\n"
                yield f"data: [EXIT 0]\n\n"
                return

        yield f"data: \n\n"
        yield f"data: Exhausted all {len(schedule)} attempts. Final WNS: {wns:+.3f} ns\n\n"
        yield f"data: Consider manually lowering CORE_UTILIZATION further or adjusting placement.\n\n"
        yield f"data: [EXIT 1]\n\n"

    return Response(_generate(), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


def _preconvert_stages(stages):
    """Background: convert any ODB stages that don't have a cached DEF yet."""
    for s in stages:
        if s.get('odb_path') and not os.path.exists(s['def_path']):
            odb_in  = s['odb_path'].replace(FLOW_DIR + '/', '/work/')
            def_out = s['def_path'].replace(FLOW_DIR + '/', '/work/')
            cmd = (
                f"util/docker_shell openroad -python /work/ui/odb_to_def.py"
                f" --odb {odb_in} --out {def_out}"
            )
            try:
                subprocess.run(cmd, shell=True, cwd=FLOW_DIR,
                               stdin=subprocess.DEVNULL,
                               capture_output=True, text=True,
                               timeout=120)
            except Exception:
                pass  # best-effort; layout_data will retry on demand


@app.route("/api/layout/stages/<platform>/<design>")
def layout_stages(platform, design):
    stages = get_available_stages(platform, design, FLOW_DIR)
    # Kick off background pre-conversion so layout requests hit the cache
    threading.Thread(target=_preconvert_stages, args=(stages,), daemon=True).start()
    return jsonify(stages)


@app.route("/api/layout/<platform>/<design>/<stage>")
def layout_data(platform, design, stage):
    stages = get_available_stages(platform, design, FLOW_DIR)
    match = next((s for s in stages if s['id'] == stage), None)
    if not match:
        return jsonify({'error': f'Stage {stage} not found'}), 404

    def_path = match['def_path']

    # If no DEF exists yet, export it from the ODB via OpenROAD inside Docker.
    # Pass stdin=DEVNULL so docker_shell's "test -t 0" check returns false and
    # it omits the -ti flags — otherwise Flask's inherited TTY causes Docker to
    # hang waiting for terminal input that never comes.
    if not os.path.exists(def_path) and match.get('odb_path'):
        odb_in  = match['odb_path'].replace(FLOW_DIR + '/', '/work/')
        def_out = def_path.replace(FLOW_DIR + '/', '/work/')
        cmd = (
            f"util/docker_shell openroad -python /work/ui/odb_to_def.py"
            f" --odb {odb_in} --out {def_out}"
        )
        try:
            result = subprocess.run(cmd, shell=True, cwd=FLOW_DIR,
                                    stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True,
                                    timeout=120)
        except subprocess.TimeoutExpired:
            return jsonify({'error': 'DEF export timed out after 120 s'}), 500
        if result.returncode != 0 or not os.path.exists(def_path):
            return jsonify({'error': f'DEF export failed: {result.stderr}'}), 500

    lef_files = find_lef_files(platform, FLOW_DIR)
    macro_sizes = parse_lef_macros(lef_files)
    data = parse_def(def_path, macro_sizes)
    if data is None:
        return jsonify({'error': 'Failed to parse DEF file'}), 500
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
