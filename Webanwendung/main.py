import tkinter as tk
from tkinter import ttk, messagebox
import math
import heapq
import numpy as np


class GridAStarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("A* Pfadsuche – ohne diagonales Ecken-Schneiden")

        self.simplified = []
        self.decomp = None


        # --- Steuerleiste ---
        control = ttk.Frame(root, padding=8)
        control.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control, text="Zeilen:").pack(side=tk.LEFT)
        self.rows_var = tk.StringVar(value="25")
        rows_entry = ttk.Entry(control, textvariable=self.rows_var, width=6)
        rows_entry.pack(side=tk.LEFT, padx=(0,8))

        ttk.Label(control, text="Spalten:").pack(side=tk.LEFT)
        self.cols_var = tk.StringVar(value="25")
        cols_entry = ttk.Entry(control, textvariable=self.cols_var, width=6)
        cols_entry.pack(side=tk.LEFT, padx=(0,8))

        ttk.Button(control, text="Raster erstellen", command=self.build_grid).pack(side=tk.LEFT, padx=(0,12))

        # Moduswahl: Hindernisse / Start / Ziel
        self.mode_var = tk.StringVar(value="block")
        modes = ttk.Frame(control)
        modes.pack(side=tk.LEFT)
        ttk.Radiobutton(modes, text="Hindernisse", value="block", variable=self.mode_var).pack(side=tk.LEFT)
        ttk.Radiobutton(modes, text="Start setzen", value="start", variable=self.mode_var).pack(side=tk.LEFT)
        ttk.Radiobutton(modes, text="Ziel setzen", value="goal", variable=self.mode_var).pack(side=tk.LEFT)

        ttk.Button(control, text="Bestätigen (A*)", command=self.run_astar).pack(side=tk.LEFT, padx=12)
        ttk.Button(control, text="Zurücksetzen", command=self.reset_all).pack(side=tk.LEFT)
        ttk.Button(control, text="Smoothing", command=self.smoothing).pack(side=tk.LEFT, padx=(12,0))
        ttk.Button(control, text="Pfad vereinfachen", command=self.simplify_path).pack(side=tk.LEFT, padx=(12,0))
        ttk.Button(control, text="Zerlegung", command=self.decompose).pack(side=tk.LEFT, padx=(6,0))



        # --- Canvas ---
        self.canvas_wrap = ttk.Frame(root)
        self.canvas_wrap.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_wrap, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bindings
        # Start/Ziel jetzt auf ButtonRelease-1, damit es sich nicht mit dem Malen (ButtonPress-1) beißt
        self.canvas.bind("<ButtonRelease-1>", self.on_click_release)
        # Malen/Radieren im Hindernis-Modus
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
        self._drag_mode = None      # None | "draw" | "erase"
        self._drag_painted = set()

        # Initiales Raster
        self.build_grid()

    # ----------------- UI / Raster -----------------
    def build_grid(self):
        try:
            self.rows = max(2, int(self.rows_var.get()))
            self.cols = max(2, int(self.cols_var.get()))
        except ValueError:
            messagebox.showerror("Fehler", "Bitte ganze Zahlen für Zeilen/Spalten eingeben.")
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
                x0 = c * self.cell_size
                y0 = r * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                rid = self.canvas.create_rectangle(x0, y0, x1, y1, outline="#ddd", fill="white", tags=("cell",))
                self.rect_ids[r][c] = rid

        for r in range(0, self.rows+1, 5):
            y = r * self.cell_size
            self.canvas.create_line(0, y, width, y, fill="#bbb", tags=("gridline",))

        for c in range(0, self.cols+1, 5):
            x = c * self.cell_size
            self.canvas.create_line(x, 0, x, height, fill="#bbb", tags=("gridline",))

        # Rand besetzen (Default) + sofort einfärben
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




    def reset_all(self):
        self.blocks.clear()
        self.start = None
        self.goal = None
        self.clear_visuals()

    def clear_visuals(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.set_cell_color((r, c), "white")
        for (r, c) in self.blocks:
            self.set_cell_color((r, c), "black")
        if self.start:
            self.set_cell_color(self.start, "#2ecc71")
        if self.goal:
            self.set_cell_color(self.goal, "#e74c3c")
        self.path = []

    # ----------------- Maushandling -----------------
    def cell_from_xy(self, x, y):
        c = x // self.cell_size
        r = y // self.cell_size
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return None
        return (r, c)

    def on_press(self, event):
        # Nur im Hindernis-Modus „malen“
        if self.mode_var.get() != "block":
            return
        cell = self.cell_from_xy(event.x, event.y)
        if cell is None or cell == self.start or cell == self.goal:
            return "break"

        # Festlegen, ob wir in diesem Drag zeichnen oder radieren
        if cell in self.blocks:
            self._drag_mode = "erase"
            self.blocks.remove(cell)
            self.set_cell_color(cell, "white")
        else:
            self._drag_mode = "draw"
            self.blocks.add(cell)
            self.set_cell_color(cell, "black")

        self._drag_painted = {cell}
        self.remove_path_only()
        return "break"  # verhindert, dass andere Button-1-Handler feuern

    def on_drag(self, event):
        if self.mode_var.get() != "block" or not self._drag_mode:
            return
        cell = self.cell_from_xy(event.x, event.y)
        if cell is None or cell in self._drag_painted or cell in (self.start, self.goal):
            return "break"

        if self._drag_mode == "draw":
            if cell not in self.blocks:
                self.blocks.add(cell)
                self.set_cell_color(cell, "black")
        else:
            if cell in self.blocks:
                self.blocks.remove(cell)
                self.set_cell_color(cell, "white")

        self._drag_painted.add(cell)
        return "break"

    def on_release(self, _event):
        if self.mode_var.get() != "block":
            return
        self._drag_mode = None
        self._drag_painted = set()


    def on_click_release(self, event):
        """Start/Ziel setzen bei Loslassen der Taste, wenn nicht im Blockmodus."""
        if self.mode_var.get() == "block":
            return
        cell = self.cell_from_xy(event.x, event.y)
        if cell is None:
            return
        if self.mode_var.get() == "start":
            if cell in self.blocks or cell == self.goal:
                return
            prev = self.start
            self.start = cell
            if prev:
                self.set_cell_color(prev, "white")
            self.set_cell_color(self.start, "#2ecc71")
        elif self.mode_var.get() == "goal":
            if cell in self.blocks or cell == self.start:
                return
            prev = self.goal
            self.goal = cell
            if prev:
                self.set_cell_color(prev, "white")
            self.set_cell_color(self.goal, "#e74c3c")
        self.remove_path_only()

    # ----------------- Mal-/Zeichenhilfen -----------------
    def set_cell_color(self, cell, color):
        r, c = cell
        rid = self.rect_ids[r][c]
        self.canvas.itemconfig(rid, fill=color)

    def draw_path(self, path):
        for (r, c) in path:
            if (r, c) != self.start and (r, c) != self.goal:
                self.set_cell_color((r, c), "#3498db")

    def remove_path_only(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.blocks and (r, c) != self.start and (r, c) != self.goal:
                    self.set_cell_color((r, c), "white")

    # ----------------- A* -----------------
    def run_astar(self):
        if self.start is None or self.goal is None:
            messagebox.showinfo("Hinweis", "Bitte Start und Ziel setzen.")
            return
        if self.start in self.blocks or self.goal in self.blocks:
            messagebox.showerror("Fehler", "Start/Ziel darf nicht auf einem Hindernis liegen.")
            return

        path = self.astar(self.start, self.goal)
        self.remove_path_only()
        if path is None:
            messagebox.showinfo("Kein Pfad", "Es konnte kein Pfad gefunden werden.")
            return
        self.path = path
        self.draw_path(self.path)

    def astar(self, start, goal):
        rows, cols = self.rows, self.cols
        blocks = self.blocks

        def in_bounds(rc):
            r, c = rc
            return 0 <= r < rows and 0 <= c < cols

        directions = [
            (-1,  0), (1,  0), (0, -1), (0,  1),      # orthogonal
            (-1, -1), (-1, 1), (1, -1), (1, 1)        # diagonal
        ]

        def neighbors(rc):
            r, c = rc
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if not in_bounds(nxt):
                    continue
                if nxt in blocks:
                    continue
                # kein diagonales Ecken-Schneiden
                if dr != 0 and dc != 0:
                    if (r + dr, c) in blocks or (r, c + dc) in blocks:
                        continue
                yield nxt

        def cost(a, b):
            (r1, c1), (r2, c2) = a, b
            if r1 == r2 or c1 == c2:
                return 1.0
            return math.sqrt(2)

        def heuristic(a, b):
            (r1, c1), (r2, c2) = a, b
            dx = abs(c1 - c2); dy = abs(r1 - r2)
            return (max(dx, dy) - min(dx, dy)) * 1.0 + min(dx, dy) * math.sqrt(2)

        open_heap = []
        g_score = {start: 0.0}
        heapq.heappush(open_heap, (heuristic(start, goal), 0, start))
        came_from = {}
        closed = set()
        pushcount = 1

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return self.reconstruct_path(came_from, current)
            closed.add(current)

            for nxt in neighbors(current):
                tentative = g_score[current] + cost(current, nxt)
                if nxt in g_score and tentative >= g_score[nxt]:
                    continue
                came_from[nxt] = current
                g_score[nxt] = tentative
                f = tentative + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f, pushcount, nxt))
                pushcount += 1

        return None

    @staticmethod
    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def _clip_polygon_halfspace(self, poly, a, b):
        """Sutherland–Hodgman-Clipping mit Halbraum a.x <= b (2D). poly in Grid-Koordinaten."""
        if not poly:
            return []
        ax, ay = float(a[0]), float(a[1])
        b = float(b)
        def inside(P):
            return ax*P[0] + ay*P[1] <= b + 1e-9
        def intersect(P, Q):
            x1, y1 = P; x2, y2 = Q
            f1 = ax*x1 + ay*y1 - b
            f2 = ax*x2 + ay*y2 - b
            if abs(f1 - f2) < 1e-12:
                return Q  # nahezu parallel
            t = f1 / (f1 - f2)
            return (x1 + t*(x2-x1), y1 + t*(y2-y1))
        out = []
        for i in range(len(poly)):
            C = poly[i]
            P = poly[i-1]
            if inside(C):
                if not inside(P):
                    out.append(intersect(P, C))
                out.append(C)
            elif inside(P):
                out.append(intersect(P, C))
        return out

    def _draw_decomposition(self, A_list, B_list):
        """Zeichnet aus A x <= b die Polygone auf den Canvas (2D, Grid-Koordinaten)."""
        self.canvas.delete("decomp")
        bbox = [
            (0.0, 0.0),
            (float(self.cols), 0.0),
            (float(self.cols), float(self.rows)),
            (0.0, float(self.rows)),
        ]
        for A, b in zip(A_list, B_list):
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





    def decompose(self):
        # Voraussetzungen
        if not self.path:
            messagebox.showinfo("Hinweis", "Bitte zuerst mit A* einen Pfad berechnen.")
            return
        if not self.simplified:
            # automatisch vereinfachen, falls noch nicht erfolgt
            self.simplify_path()
            if not self.simplified:
                return

        try:
            import pydecomp as pdc  # erwartet: pdc.convex_decomposition_2D
        except Exception as e:
            messagebox.showerror("Modul fehlt", f"Das Modul 'pdc' konnte nicht importiert werden:\n{e}")
            return

        # Hindernisse: Zentren der belegten Zellen (Grid-Koordinaten)
        obstacles = np.array([[c + 0.5, r + 0.5] for (r,c) in sorted(self.blocks)], dtype=float)
        # Pfad (vereinfacht) in Grid-Koordinaten
        path_real = np.array([[c + 0.5, r + 0.5] for (r,c) in self.simplified], dtype=float)
        # Arbeitsbereich (Breite, Höhe) in Grid-Einheiten
        area_size = np.array([[float(self.cols), float(self.rows)]], dtype=float)

        try:
            A_list, b_list = pdc.convex_decomposition_2D(obstacles, path_real, area_size)
        except Exception as e:
            messagebox.showerror("Zerlegung fehlgeschlagen", str(e))
            return

        self.decomp = (A_list, b_list)
        # Flächen einzeichnen
        self._draw_decomposition(A_list, b_list)

    
    def smoothing(self):
        # Voraussetzung: ein A*-Pfad existiert
        if not self.path:
            messagebox.showinfo("Hinweis", "Bitte zuerst mit A* einen Pfad berechnen.")
            return

        # Alte Darstellung entfernen/verdecken
        # 1) Gitterlinien ausblenden
        self.canvas.delete("gridline")
        # 2) Pfad-Färbung zurücksetzen (blau weg), Start/Ziel beibehalten
        self.remove_path_only()
        # 3) Zellenumrandungen entfernen, damit das Raster „verschwindet“
        for r in range(self.rows):
            for c in range(self.cols):
                rid = self.rect_ids[r][c]
                # outline entfernen
                self.canvas.itemconfig(rid, outline="")

        # Bereits vorhandene Bäume/Trajektorie löschen (falls schon einmal gedrückt)
        self.canvas.delete("tree")
        self.canvas.delete("traj")
        self.canvas.delete("marker")

        # Aus Hindernissen „grüne Bäume“ machen
        # (einfach: Zelle grün färben + kleiner Stamm + Krone als Oval)
        for (r, c) in self.blocks:
            x0 = c * self.cell_size
            y0 = r * self.cell_size
            x1 = x0 + self.cell_size
            y1 = y0 + self.cell_size
            # Zelle grün
            # Zelle weiß
            self.set_cell_color((r, c), "white")
            # kleiner brauner Stamm
            sx0 = x0 + self.cell_size*0.45
            sx1 = x0 + self.cell_size*0.55
            sy0 = y0 + self.cell_size*0.60
            sy1 = y0 + self.cell_size*0.90
            self.canvas.create_rectangle(sx0, sy0, sx1, sy1, fill="#8e5a2b", width=0, tags=("tree",))
            # grüne Krone
            cx0 = x0 + self.cell_size*0.20
            cy0 = y0 + self.cell_size*0.10
            cx1 = x0 + self.cell_size*0.80
            cy1 = y0 + self.cell_size*0.70
            self.canvas.create_oval(cx0, cy0, cx1, cy1, fill="#27ae60", width=0, tags=("tree",))

        # Punkte aus dem A*-Pfad in Pixelzentren umrechnen
        pts = []
        half = self.cell_size / 2
        for (r, c) in self.path:
            x = c * self.cell_size + half
            y = r * self.cell_size + half
            pts.append((x, y))

        # Optional: doppelte aufeinanderfolgende Punkte entfernen
        dedup = [pts[0]]
        for p in pts[1:]:
            if p != dedup[-1]:
                dedup.append(p)
        pts = dedup

        # „Hübsche“ Trajektorie: Canvas-Spline zeichnen
        # smooth=True nutzt eine Spline-Interpolation. splinesteps erhöht die Glätte.
        flat = [coord for xy in pts for coord in xy]
        if len(pts) >= 2:
            self.canvas.create_line(
                *flat, fill="#e67e22", width=3, smooth=True, splinesteps=24, tags=("traj",)
            )

        # Start/Ziel als Marker hervorheben (Kreise)
        for cell, color in ((self.start, "#2ecc71"), (self.goal, "#e74c3c")):
            if cell:
                cx = cell[1] * self.cell_size + half
                cy = cell[0] * self.cell_size + half
                r = self.cell_size * 0.30
                self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline=color, width=3, tags=("marker",))

    def simplify_path(self, epsilon=1.3):
        """
        Vereinfacht den aktuellen Pfad mit dem Ramer–Douglas–Peucker-Algorithmus.
        epsilon = Toleranz in Grid-Zellen.
        """
        if not self.path or len(self.path) < 2:
            messagebox.showinfo("Hinweis", "Bitte erst einen Pfad berechnen.")
            return

        pts = np.array([[c, r] for (r, c) in self.path], dtype=float)

        def rdp(points, eps):
            if len(points) < 3:
                return points
            # maximale Abweichung zur Verbindungslinie berechnen
            start, end = points[0], points[-1]
            line_vec = end - start
            line_len = np.linalg.norm(line_vec)
            if line_len == 0:
                return np.vstack([start, end])
            dists = np.abs(np.cross(line_vec, points[1:-1] - start)) / line_len
            idx = np.argmax(dists)
            if dists[idx] > eps:
                left = rdp(points[: idx+2], eps)
                right = rdp(points[idx+1:], eps)
                return np.vstack([left[:-1], right])
            else:
                return np.vstack([start, end])

        simp = rdp(pts, epsilon)
        simp = simp.astype(int)  # wieder in Integer-Rasterkoordinaten

        # Speichern & visualisieren
        self.simplified = [(int(r), int(c)) for (c, r) in simp]

        half = self.cell_size/2
        self.canvas.delete("simp")
        for (r,c) in self.simplified:
            x = c*self.cell_size + half
            y = r*self.cell_size + half
            self.canvas.create_oval(x-4,y-4,x+4,y+4, fill="orange", tags=("simp",))


if __name__ == "__main__":
    root = tk.Tk()
    app = GridAStarApp(root)
    root.mainloop()
