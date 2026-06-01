"""
Flask web UI for OpenROAD-flow-scripts.

Run from the flow/ directory:
    python3 ui/app.py

Then open http://localhost:5000 in a browser.
"""

import glob
import os
import subprocess
import sys

import torch

from flask import Flask, Response, jsonify, render_template, request, send_file
from layout_parser import (
    find_lef_files, parse_lef_macros, get_available_stages,
    get_design_nickname, parse_def,
)

app = Flask(__name__)

FLOW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    try:
        model = torch.compile(model)
    except Exception:
        pass  # torch.compile is optional; skip if unsupported
    _MODEL_CACHE[checkpoint] = model
    return model


def stream_command(cmd):
    """Run a shell command and yield output lines as SSE messages."""
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=FLOW_DIR,
    )
    for line in iter(process.stdout.readline, ""):
        yield f"data: {line.rstrip()}\n\n"
    process.wait()
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
    cmd = f"util/docker_shell make DESIGN_CONFIG={config} DESIGN_HOME=/work/designs"
    return Response(stream_command(cmd), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


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


@app.route("/api/floorplan/tcl/<platform>/<design>")
def get_floorplan_tcl(platform, design):
    tcl_path = os.path.join(FLOW_DIR, f"ml/data/{platform}_{design}_suggested.tcl")
    if os.path.exists(tcl_path):
        with open(tcl_path) as f:
            return f.read(), 200, {"Content-Type": "text/plain"}
    return "Tcl file not found — run Suggest Floorplan first.", 404


@app.route("/api/layout/stages/<platform>/<design>")
def layout_stages(platform, design):
    stages = get_available_stages(platform, design, FLOW_DIR)
    return jsonify(stages)


@app.route("/api/layout/<platform>/<design>/<stage>")
def layout_data(platform, design, stage):
    stages = get_available_stages(platform, design, FLOW_DIR)
    match = next((s for s in stages if s['id'] == stage), None)
    if not match:
        return jsonify({'error': f'Stage {stage} not found'}), 404

    def_path = match['def_path']

    # If no DEF exists yet, export it from the ODB via OpenROAD inside Docker
    if not os.path.exists(def_path) and match.get('odb_path'):
        odb_in  = match['odb_path'].replace(FLOW_DIR + '/', '/work/')
        def_out = def_path.replace(FLOW_DIR + '/', '/work/')
        cmd = (
            f"util/docker_shell openroad -python /work/ui/odb_to_def.py"
            f" --odb {odb_in} --out {def_out}"
        )
        result = subprocess.run(cmd, shell=True, cwd=FLOW_DIR,
                                capture_output=True, text=True)
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
