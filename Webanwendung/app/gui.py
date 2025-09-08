import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from core.astar import astar
from core.simplify import simplify_rdp
from core.decomposition import convex_decompose_and_clip


class GridAStarApp:
    def __init__(self, root):

        style = ttk.Style()
        style.theme_use("calm")


        self.root = root
        self.root.title("A* Pfadsuche – ohne diagonales Ecken-Schneiden")

        self.simplified = []
        self.decomp = None

        # --- Steuerleiste ---
        control = ttk.Frame(root, padding=8)
        control.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control, text="Zeilen:").pack(side=tk.LEFT)
        self.rows_var = tk.StringVar(value="25")
        ttk.Entry(control, textvariable=self.rows_var, width=6).pack(side=tk.LEFT, padx=(0,8))

        ttk.Label(control, text="Spalten:").pack(side=tk.LEFT)
        self.cols_var = tk.StringVar(value="25")
        ttk.Entry(control, textvariable=self.cols_var, width=6).pack(side=tk.LEFT, padx=(0,8))

        ttk.Button(control, text="Raster erstellen", command=self.build_grid).pack(side=tk.LEFT, padx=(0,12))

        # Moduswahl: Hindernisse / Start / Ziel
        self.mode_var = tk.StringVar(value="block")
        modes = ttk.Frame(control)
        modes.pack(side=tk.LEFT)
        ttk.Radiobutton(modes, text="Hindernisse", value="block", variable=self.mode_var).pack(side=tk.LEFT)
        ttk.Radiobutton(modes, text="Start setzen", value="start", variable=self.mode_var).pack(side=tk.LEFT)
        ttk.Radiobutton(modes, text="Ziel setzen", value="goal", variable=self.mode_var).pack(side=tk.LEFT)

        ttk.Button(control, text="Bestätigen (A*)", command=self.run_astar).pack(side=tk.LEFT, padx=12)
        ttk.Button(control, text="Smoothing", command=self.smoothing).pack(side=tk.LEFT, padx=(12,0))
        ttk.Button(control, text="Pfad vereinfachen", command=self.simplify_path).pack(side=tk.LEFT, padx=(12,0))
        ttk.Button(control, text="Zerlegung", command=self.decompose).pack(side=tk.LEFT, padx=(6,0))
        ttk.Button(control, text="Zurücksetzen", command=self.reset_all).pack(side=tk.LEFT)


        # --- Canvas ---
        self.canvas_wrap = ttk.Frame(root)
        self.canvas_wrap.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_wrap, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bindings
        self.canvas.bind("<ButtonRelease-1>", self.on_click_release)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release, add="+")

        # Zustände
        self.cell_size = 24
        self.rows = 0
        self.cols = 0
        self.rect_ids = []
        self.blocks = set()
        self.start = None
        self.goal = None
        self.path = []

        # Drag-Malen-Zustand
        self._drag_mode = None
        self._drag_painted = set()

        self.build_grid()

    # --- Grid ---
    def build_grid(self):
        try:
            self.rows = max(2, int(self.rows_var.get()))
            self.cols = max(2, int(self.cols_var.get()))
        except ValueError:
            messagebox.showerror("Fehler", "Bitte ganze Zahlen eingeben.")
            return

        self.blocks.clear()
        self.start = None
        self.goal = None
        self.path = []

        width = self.cols * self.cell_size + 1
        height = self.rows * self.cell_size + 1
        self.canvas.config(width=width, height=height)
        self.canvas.delete("all")
        self.rect_ids = [[None for _ in range(self.cols)] for _ in range(self.rows)]

        for r in range(self.rows):
            for c in range(self.cols):
                x0, y0 = c*self.cell_size, r*self.cell_size
                x1, y1 = x0+self.cell_size, y0+self.cell_size
                rid = self.canvas.create_rectangle(
                    x0, y0, x1, y1, outline="#ddd", fill="white", tags=("cell",)
                )
                self.rect_ids[r][c] = rid

        for r in range(0, self.rows+1, 5):
            self.canvas.create_line(0, r*self.cell_size, width, r*self.cell_size, fill="#bbb", tags=("gridline",))
        for c in range(0, self.cols+1, 5):
            self.canvas.create_line(c*self.cell_size, 0, c*self.cell_size, height, fill="#bbb", tags=("gridline",))

        # Rand belegen
        for r in range(self.rows):
            self.blocks.add((r, 0))
            self.set_cell_color((r, 0), "black")
            self.blocks.add((r, self.cols-1))
            self.set_cell_color((r, self.cols-1), "black")
        for c in range(self.cols):
            self.blocks.add((0, c))
            self.set_cell_color((0, c), "black")
            self.blocks.add((self.rows-1, c))
            self.set_cell_color((self.rows-1, c), "black")

    # --- Reset ---
    def reset_all(self):
        self.blocks.clear()
        self.start = None
        self.goal = None
        self.path = []
        self.build_grid()

    # --- Hilfen ---
    def set_cell_color(self, cell, color):
        r, c = cell
        rid = self.rect_ids[r][c]
        self.canvas.itemconfig(rid, fill=color)

    def cell_from_xy(self, x, y):
        c, r = x // self.cell_size, y // self.cell_size
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return None
        return (r, c)

    # --- Maussteuerung ---
    def on_press(self, event):
        if self.mode_var.get() != "block":
            return
        cell = self.cell_from_xy(event.x, event.y)
        if not cell or cell in (self.start, self.goal):
            return "break"
        if cell in self.blocks:
            self._drag_mode = "erase"
            self.blocks.remove(cell)
            self.set_cell_color(cell, "white")
        else:
            self._drag_mode = "draw"
            self.blocks.add(cell)
            self.set_cell_color(cell, "black")
        self._drag_painted = {cell}
        return "break"

    def on_drag(self, event):
        if self.mode_var.get() != "block" or not self._drag_mode:
            return
        cell = self.cell_from_xy(event.x, event.y)
        if not cell or cell in self._drag_painted or cell in (self.start, self.goal):
            return "break"
        if self._drag_mode == "draw":
            self.blocks.add(cell)
            self.set_cell_color(cell, "black")
        else:
            self.blocks.discard(cell)
            self.set_cell_color(cell, "white")
        self._drag_painted.add(cell)
        return "break"

    def on_release(self, _):
        if self.mode_var.get() != "block":
            return
        self._drag_mode = None
        self._drag_painted.clear()

    def on_click_release(self, event):
        if self.mode_var.get() == "block":
            return
        cell = self.cell_from_xy(event.x, event.y)
        if not cell:
            return
        if self.mode_var.get() == "start" and cell not in self.blocks:
            if self.start: self.set_cell_color(self.start, "white")
            self.start = cell
            self.set_cell_color(cell, "#2ecc71")
        elif self.mode_var.get() == "goal" and cell not in self.blocks:
            if self.goal: self.set_cell_color(self.goal, "white")
            self.goal = cell
            self.set_cell_color(cell, "#e74c3c")

    # --- A* ---
    def run_astar(self):
        if not self.start or not self.goal:
            messagebox.showinfo("Hinweis", "Bitte Start und Ziel setzen.")
            return
        self.path = astar(self.rows, self.cols, self.blocks, self.start, self.goal)
        if not self.path:
            messagebox.showinfo("Kein Pfad", "Es konnte kein Pfad gefunden werden.")
            return
        for (r, c) in self.path:
            if (r, c) not in (self.start, self.goal):
                self.set_cell_color((r, c), "#3498db")

        # --- Smoothing ---
    def smoothing(self):
        if not self.path:
            messagebox.showinfo("Hinweis", "Bitte zuerst mit A* einen Pfad berechnen.")
            return

        # Gitterlinien ausblenden
        self.canvas.delete("gridline")

        # Pfadfarbe zurücksetzen
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.blocks and (r, c) not in (self.start, self.goal):
                    self.set_cell_color((r, c), "white")

        # Zellenumrandungen entfernen
        for r in range(self.rows):
            for c in range(self.cols):
                rid = self.rect_ids[r][c]
                self.canvas.itemconfig(rid, outline="")

        # Alte Objekte löschen
        self.canvas.delete("tree")
        self.canvas.delete("traj")
        self.canvas.delete("marker")

        # Hindernisse → Bäume
        for (r, c) in self.blocks:
            x0, y0 = c*self.cell_size, r*self.cell_size
            x1, y1 = x0+self.cell_size, y0+self.cell_size

            # Hintergrund weiß halten
            self.set_cell_color((r, c), "white")

            # Stamm
            sx0, sx1 = x0+self.cell_size*0.45, x0+self.cell_size*0.55
            sy0, sy1 = y0+self.cell_size*0.60, y0+self.cell_size*0.90
            self.canvas.create_rectangle(sx0, sy0, sx1, sy1, fill="#8e5a2b", width=0, tags=("tree",))

            # Krone
            cx0, cx1 = x0+self.cell_size*0.20, x0+self.cell_size*0.80
            cy0, cy1 = y0+self.cell_size*0.10, y0+self.cell_size*0.70
            self.canvas.create_oval(cx0, cy0, cx1, cy1, fill="#27ae60", width=0, tags=("tree",))

        # Pfadpunkte in Pixel-Koordinaten
        half = self.cell_size/2
        pts = [(c*self.cell_size+half, r*self.cell_size+half) for (r, c) in self.path]

        # Duplikate entfernen
        dedup = [pts[0]]
        for p in pts[1:]:
            if p != dedup[-1]:
                dedup.append(p)
        pts = dedup

        # Glatte Trajektorie
        if len(pts) >= 2:
            flat = [coord for xy in pts for coord in xy]
            self.canvas.create_line(
                *flat, fill="#e67e22", width=3,
                smooth=True, splinesteps=24, tags=("traj",)
            )

        # Start/Ziel als Marker
        for cell, color in ((self.start, "#2ecc71"), (self.goal, "#e74c3c")):
            if cell:
                cx, cy = cell[1]*self.cell_size+half, cell[0]*self.cell_size+half
                r = self.cell_size*0.30
                self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline=color, width=3, tags=("marker",))


    # --- Vereinfachung ---
    def simplify_path(self):
        if not self.path:
            messagebox.showinfo("Hinweis", "Bitte erst A* laufen lassen.")
            return
        simp = simplify_rdp(self.path, epsilon=1.3)
        self.simplified = simp
        self.canvas.delete("simp")
        half = self.cell_size/2
        for (r, c) in simp:
            x, y = c*self.cell_size+half, r*self.cell_size+half
            self.canvas.create_oval(x-4,y-4,x+4,y+4, fill="orange", tags=("simp",))

    # --- Zerlegung ---
    def decompose(self):
        if not self.simplified:
            self.simplify_path()
        if not self.simplified:
            return
        try:
            A_list, b_list = convex_decompose_and_clip(self.blocks, self.simplified, self.cols, self.rows)
        except Exception as e:
            messagebox.showerror("Fehler bei Zerlegung", str(e))
            return
    
        self.canvas.delete("decomp")
        bbox = [
            (0.0, 0.0),
            (float(self.cols), 0.0),
            (float(self.cols), float(self.rows)),
            (0.0, float(self.rows)),
        ]
        for A, b in zip(A_list, b_list):
            A = np.asarray(A, dtype=float)
            b = np.asarray(b, dtype=float).reshape(-1)
            if A.size == 0:
                continue
            poly = bbox[:]
            for k in range(A.shape[0]):
                poly = self._clip_polygon_halfspace(poly, (A[k, 0], A[k, 1]), b[k])
                if not poly:
                    break
            if poly and len(poly) >= 3:
                coords = []
                for x, y in poly:
                    px = x * self.cell_size
                    py = y * self.cell_size
                    coords += [px, py]
                self.canvas.create_polygon(
                    *coords,
                    fill="#9ec9ff",
                    stipple="gray50",   # wirkt wie Transparenz
                    outline="#2c7be5",
                    width=2,
                    tags=("decomp",)
                )

        # Nach dem Zeichnen alle Flächen korrekt einsortieren:
        # über den Zellen & Grid …
        self.canvas.tag_raise("decomp", "cell")
        self.canvas.tag_raise("decomp", "gridline")
        # … aber unter Bäumen/Trajektorie/Marker/Simplify-Punkten
        for top_tag in ("tree", "traj", "marker", "simp"):
            try:
                self.canvas.tag_lower("decomp", top_tag)
            except tk.TclError:
                pass

    def _clip_polygon_halfspace(self, poly, a, b):
        if not poly: return []
        ax, ay, b = float(a[0]), float(a[1]), float(b)
        def inside(P): return ax*P[0]+ay*P[1] <= b+1e-9
        def intersect(P,Q):
            f1, f2 = ax*P[0]+ay*P[1]-b, ax*Q[0]+ay*Q[1]-b
            if abs(f1-f2)<1e-12: return Q
            t=f1/(f1-f2)
            return (P[0]+t*(Q[0]-P[0]), P[1]+t*(Q[1]-P[1]))
        out=[]
        for i in range(len(poly)):
            C,P=poly[i],poly[i-1]
            if inside(C):
                if not inside(P): out.append(intersect(P,C))
                out.append(C)
            elif inside(P):
                out.append(intersect(P,C))
        return out
