from flask import Flask, request, jsonify, render_template
import numpy as np

# --- your core modules ---
from core.astar import astar
from core.simplify import simplify_rdp
from core.decomposition import convex_decompose_and_clip
# use the full ADMM you wired earlier
from core.admm import run_admm_from_app

app = Flask(__name__, template_folder="templates", static_folder="static")

# --------- helpers ----------
def _clip_polygon_halfspace(poly, a, b):
    if not poly: return []
    ax, ay = float(a[0]), float(a[1]); b = float(b)
    def inside(P): return ax*P[0] + ay*P[1] <= b + 1e-9
    def intersect(P, Q):
        f1 = ax*P[0] + ay*P[1] - b
        f2 = ax*Q[0] + ay*Q[1] - b
        if abs(f1 - f2) < 1e-12: return Q
        t = f1 / (f1 - f2)
        return (P[0] + t*(Q[0]-P[0]), P[1] + t*(Q[1]-P[1]))
    out = []
    for i in range(len(poly)):
        C, P = poly[i], poly[i-1]
        if inside(C):
            if not inside(P): out.append(intersect(P, C))
            out.append(C)
        elif inside(P):
            out.append(intersect(P, C))
    return out

def _polygon_centroid(poly):
    # simple polygon centroid; poly is list of (x,y)
    if not poly: return (0.0, 0.0)
    A = 0.0; cx = 0.0; cy = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % len(poly)]
        cross = x1*y2 - x2*y1
        A += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    A *= 0.5
    if abs(A) < 1e-9:  # fallback: average
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        return (float(np.mean(xs)), float(np.mean(ys)))
    cx /= (6.0*A); cy /= (6.0*A)
    return (cx, cy)

# --------- routes ----------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run_astar", methods=["POST"])
def run_astar_api():
    data = request.json
    rows = int(data["rows"])
    cols = int(data["cols"])
    blocks = set(tuple(b) for b in data["blocks"])  # [(r,c), ...]
    start = tuple(data["start"])
    goal = tuple(data["goal"])
    path = astar(rows, cols, blocks, start, goal) or []
    return jsonify(path)

@app.route("/simplify", methods=["POST"])
def simplify_api():
    data = request.json
    path = [tuple(p) for p in data["path"]]
    epsilon = float(data.get("epsilon", 1.3))
    simp = simplify_rdp(path, epsilon=epsilon) or []
    return jsonify(simp)

@app.route("/decompose", methods=["POST"])
def decompose_api():
    data = request.json
    rows = int(data["rows"])
    cols = int(data["cols"])
    blocks = set(tuple(b) for b in data["blocks"])
    simplified = [tuple(p) for p in data["simplified"]]

    if not simplified:
        return jsonify({"error": "Simplified path is empty"}), 400

    # compute half-space sets
    A_list, b_list = convex_decompose_and_clip(blocks, simplified, cols, rows)

    # turn each (A,b) into a polygon by clipping the canvas bbox
    bbox = [(0.0,0.0), (float(cols),0.0), (float(cols),float(rows)), (0.0,float(rows))]
    regions = []
    for k, (A, b) in enumerate(zip(A_list, b_list)):
        A = np.asarray(A, float)
        b = np.asarray(b, float).reshape(-1)
        if A.size == 0: 
            continue
        poly = bbox[:]
        for i in range(A.shape[0]):
            poly = _clip_polygon_halfspace(poly, (A[i,0], A[i,1]), b[i])
            if not poly: break
        if poly and len(poly) >= 3:
            cx, cy = _polygon_centroid(poly)
            regions.append({
                "id": k,
                "poly": [(float(x), float(y)) for (x,y) in poly],
                "centroid": (float(cx), float(cy))
            })

    # return regions for visualization + raw A,b for ADMM
    A_out = [np.asarray(A, float).tolist() for A in A_list]
    B_out = [np.asarray(b, float).reshape(-1).tolist() for b in b_list]
    return jsonify({"regions": regions, "A_list": A_out, "b_list": B_out})

@app.route("/run_admm", methods=["POST"])
def run_admm_api():
    data = request.json
    import numpy as np

    A_list = [np.asarray(A, float) for A in data["A_list"]]
    b_list = [np.asarray(b, float) for b in data["b_list"]]
    simplified = [tuple(c) for c in data["simplified"]]
    start = tuple(data["start"])
    goal = tuple(data["goal"])
    cols = int(data["cols"])
    rows = int(data["rows"])

    # optional capture params from client (fallbacks are fine)
    capture = bool(data.get("capture", True))
    capture_every = int(data.get("capture_every", 1))
    iter_plot_samples = int(data.get("iter_plot_samples", 12))

    result = run_admm_from_app(
        A_list, b_list,
        simplified_cells=simplified,
        start_cell=start, goal_cell=goal,
        grid_cols=cols, grid_rows=rows,
        config_path="config/admm.json",
        capture_iterations=capture,
        capture_every=capture_every,
        iter_plot_samples=iter_plot_samples,
    )

    # numpy -> python
    def to_py(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, list):
            return [to_py(x) for x in obj]
        if isinstance(obj, tuple):
            return [to_py(x) for x in obj]
        if isinstance(obj, dict):
            return {k: to_py(v) for k, v in obj.items()}
        return obj

    return jsonify(to_py(result))

if __name__ == "__main__":
    app.run(debug=True)
