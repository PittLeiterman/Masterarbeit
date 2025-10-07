#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
import math

# 3D plotting (embedded in Tkinter)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# ---------- Voxel helpers ----------
def clamp_int(v, lo, hi):
    return max(lo, min(int(v), hi))

def voxelize_cylinder(center_x, center_y, base_z, radius, height, nx, ny, nz, voxels):
    """
    Solid vertical cylinder on voxel grid:
      - axis along +Z
      - inclusive base_z .. base_z + height - 1
      - circle test in XY using integer voxel centers (x, y) at each z layer
    """
    if radius <= 0 or height <= 0:
        return
    x0 = int(round(center_x))
    y0 = int(round(center_y))
    z0 = int(round(base_z))
    r = int(radius)
    r2 = r * r

    x_min = clamp_int(x0 - r, 0, nx - 1)
    x_max = clamp_int(x0 + r, 0, nx - 1)
    y_min = clamp_int(y0 - r, 0, ny - 1)
    y_max = clamp_int(y0 + r, 0, ny - 1)
    z_min = clamp_int(z0, 0, nz - 1)
    z_max = clamp_int(z0 + int(height) - 1, 0, nz - 1)

    for z in range(z_min, z_max + 1):
        for x in range(x_min, x_max + 1):
            dx2 = (x - x0) * (x - x0)
            rem = r2 - dx2
            if rem < 0:
                continue
            y_half = int(math.isqrt(rem))
            yy_min = max(y0 - y_half, y_min)
            yy_max = min(y0 + y_half, y_max)
            for y in range(yy_min, yy_max + 1):
                voxels.add((x, y, z))

def build_forest_voxels(trees, nx, ny, nz):
    voxels = set()
    for (x, y, z0, radius, height) in trees:
        voxelize_cylinder(x, y, z0, radius, height, nx, ny, nz, voxels)
    return voxels

def build_box_shell(nx, ny, nz, thickness=1):
    """
    Return a set of voxels forming the *shell* of the axis-aligned box [0..nx-1]×[0..ny-1]×[0..nz-1]
    with the given thickness (in voxels).
    """
    t = max(1, int(thickness))
    vox = set()
    # x-faces
    for x in range(0, min(t, nx)):
        for y in range(ny):
            for z in range(nz):
                vox.add((x, y, z))
    for x in range(max(0, nx - t), nx):
        for y in range(ny):
            for z in range(nz):
                vox.add((x, y, z))
    # y-faces
    for y in range(0, min(t, ny)):
        for x in range(nx):
            for z in range(nz):
                vox.add((x, y, z))
    for y in range(max(0, ny - t), ny):
        for x in range(nx):
            for z in range(nz):
                vox.add((x, y, z))
    # z-faces
    for z in range(0, min(t, nz)):
        for x in range(nx):
            for y in range(ny):
                vox.add((x, y, z))
    for z in range(max(0, nz - t), nz):
        for x in range(nx):
            for y in range(ny):
                vox.add((x, y, z))
    return vox

# ---------- App ----------
class TreeSpawnerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tree Spawner (Cylinders on a Voxel Map)")
        self.geometry("1020x760")

        # map size
        self.nx_var = tk.IntVar(value=100)
        self.ny_var = tk.IntVar(value=100)
        self.nz_var = tk.IntVar(value=40)

        # bounds options
        self.include_bounds_export_var = tk.BooleanVar(value=True)
        self.include_bounds_preview_var = tk.BooleanVar(value=False)
        self.bound_thickness_var = tk.IntVar(value=1)

        # tree inputs
        self.tx_var = tk.IntVar(value=50)
        self.ty_var = tk.IntVar(value=50)
        self.tz_var = tk.IntVar(value=0)
        self.tr_var = tk.IntVar(value=3)
        self.th_var = tk.IntVar(value=15)

        # RANDOM spawn control
        self.rand_n_var = tk.IntVar(value=20)

        # preview controls
        self.max_preview_var = tk.IntVar(value=50000)
        self.alpha_var = tk.DoubleVar(value=0.18)
        self.tree_point_size_var = tk.DoubleVar(value=5.0)

        self.trees = []   # list of (x, y, z0, radius, height)
        self.voxels = set()           # tree voxels only (built)
        self.bounds_cache = set()     # cached bounds shell for preview/export

        self.create_widgets()
        self.init_plot()

    def create_widgets(self):
        # Map & tree inputs
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        frm_map = ttk.LabelFrame(top, text="Map Size (voxels) & Bounds")
        frm_map.pack(side="left", padx=(0,10))
        ttk.Label(frm_map, text="NX").grid(row=0, column=0, padx=5, pady=4, sticky="e")
        ttk.Entry(frm_map, width=6, textvariable=self.nx_var).grid(row=0, column=1, padx=2, pady=4)
        ttk.Label(frm_map, text="NY").grid(row=0, column=2, padx=5, pady=4, sticky="e")
        ttk.Entry(frm_map, width=6, textvariable=self.ny_var).grid(row=0, column=3, padx=2, pady=4)
        ttk.Label(frm_map, text="NZ").grid(row=0, column=4, padx=5, pady=4, sticky="e")
        ttk.Entry(frm_map, width=6, textvariable=self.nz_var).grid(row=0, column=5, padx=2, pady=4)

        ttk.Checkbutton(frm_map, text="Include bounds in export",
                        variable=self.include_bounds_export_var).grid(row=1, column=0, columnspan=3, sticky="w", padx=4)
        ttk.Checkbutton(frm_map, text="Show bounds in preview",
                        variable=self.include_bounds_preview_var,
                        command=self.update_plot).grid(row=1, column=3, columnspan=3, sticky="w", padx=4)

        ttk.Label(frm_map, text="Bound thickness").grid(row=2, column=0, padx=5, pady=4, sticky="e")
        ttk.Spinbox(frm_map, from_=1, to=20, textvariable=self.bound_thickness_var, width=6,
                    command=self._rebuild_bounds).grid(row=2, column=1, padx=2, pady=4, sticky="w")

        ttk.Button(frm_map, text="Apply Size", command=self.on_apply_size).grid(row=0, column=6, padx=8, pady=4)

        frm_tree = ttk.LabelFrame(top, text="Add Tree (cylinder)")
        frm_tree.pack(side="left", padx=(0,10))
        r = 0
        ttk.Label(frm_tree, text="X").grid(row=r, column=0, padx=4, pady=4, sticky="e")
        ttk.Entry(frm_tree, width=7, textvariable=self.tx_var).grid(row=r, column=1, padx=2, pady=4)
        ttk.Label(frm_tree, text="Y").grid(row=r, column=2, padx=4, pady=4, sticky="e")
        ttk.Entry(frm_tree, width=7, textvariable=self.ty_var).grid(row=r, column=3, padx=2, pady=4)
        ttk.Label(frm_tree, text="Z0").grid(row=r, column=4, padx=4, pady=4, sticky="e")
        ttk.Entry(frm_tree, width=7, textvariable=self.tz_var).grid(row=r, column=5, padx=2, pady=4)
        ttk.Label(frm_tree, text="Radius").grid(row=r, column=6, padx=4, pady=4, sticky="e")
        ttk.Entry(frm_tree, width=7, textvariable=self.tr_var).grid(row=r, column=7, padx=2, pady=4)
        ttk.Label(frm_tree, text="Height").grid(row=r, column=8, padx=4, pady=4, sticky="e")
        ttk.Entry(frm_tree, width=7, textvariable=self.th_var).grid(row=r, column=9, padx=2, pady=4)
        ttk.Button(frm_tree, text="Add Tree", command=self.add_tree).grid(row=r, column=10, padx=8, pady=4)

        # NEW: Random spawn controls
        ttk.Label(frm_tree, text="N random").grid(row=r+1, column=0, padx=4, pady=(2,6), sticky="e")
        ttk.Spinbox(frm_tree, from_=1, to=100000, width=7, textvariable=self.rand_n_var)\
            .grid(row=r+1, column=1, padx=2, pady=(2,6), sticky="w")
        ttk.Button(frm_tree, text="Spawn Random", command=self.spawn_random)\
            .grid(row=r+1, column=2, columnspan=2, padx=6, pady=(2,6), sticky="w")

        # Tree list + actions
        frm_list = ttk.LabelFrame(self, text="Trees (x, y, z0, radius, height)")
        frm_list.pack(fill="both", expand=False, padx=10, pady=(0,10))
        self.listbox = tk.Listbox(frm_list, height=7, font=("Consolas", 10), exportselection=False)
        self.listbox.pack(fill="both", expand=True, padx=8, pady=8)

        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=10, pady=6)
        ttk.Button(actions, text="Remove Selected", command=self.remove_selected).pack(side="left", padx=4)
        ttk.Button(actions, text="Clear Trees", command=self.clear_trees).pack(side="left", padx=4)
        ttk.Button(actions, text="Build & Preview", command=self.on_build).pack(side="left", padx=8)
        ttk.Button(actions, text="Export .txt", command=self.on_export).pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="Map not built yet.")
        ttk.Label(actions, textvariable=self.status_var).pack(side="right")

        # Preview controls
        frm_ctl = ttk.LabelFrame(self, text="3D Preview Controls")
        frm_ctl.pack(fill="x", padx=10, pady=6)
        ttk.Label(frm_ctl, text="Max voxels to draw").grid(row=0, column=0, padx=6, pady=4, sticky="e")
        ttk.Spinbox(frm_ctl, from_=2000, to=300000, increment=1000,
                    textvariable=self.max_preview_var, width=10,
                    command=self.update_plot).grid(row=0, column=1, padx=4, pady=4, sticky="w")
        ttk.Label(frm_ctl, text="Alpha").grid(row=0, column=2, padx=(18,6), pady=4, sticky="e")
        ttk.Spinbox(frm_ctl, from_=0.05, to=1.0, increment=0.05,
                    textvariable=self.alpha_var, width=6,
                    command=self.update_plot).grid(row=0, column=3, padx=4, pady=4, sticky="w")
        ttk.Label(frm_ctl, text="Voxel marker size").grid(row=0, column=4, padx=(18,6), pady=4, sticky="e")
        ttk.Spinbox(frm_ctl, from_=1.0, to=20.0, increment=1.0,
                    textvariable=self.tree_point_size_var, width=6,
                    command=self.update_plot).grid(row=0, column=5, padx=4, pady=4, sticky="w")
        ttk.Button(frm_ctl, text="Refresh 3D", command=lambda: self.update_plot(redraw_only=True)).grid(row=0, column=6, padx=(18,6), pady=4)

        # 3D plot area
        frm_plot = ttk.LabelFrame(self, text="3D Preview (drag to rotate, scroll to zoom)")
        frm_plot.pack(fill="both", expand=True, padx=10, pady=(0,10))
        self.frm_plot = frm_plot

        # Info
        info = ttk.LabelFrame(self, text="Notes")
        info.pack(fill="x", padx=10, pady=(0,10))
        tk.Label(info, justify="left", text=(
            "- Map is a voxel box of size NX × NY × NZ.\n"
            "- Trees are vertical solid cylinders at (x, y) from z=z0 up to z0+height-1.\n"
            "- 'Spawn Random' drops N trees with the current Z0/Radius/Height.\n"
            "- Export writes occupied voxels as lines: 'x y z'.\n"
            "- If enabled, the map’s bounding shell voxels (thickness T) are included in export and/or preview."
        )).pack(fill="x", padx=8, pady=8)

    # 3D setup
    def init_plot(self):
        self.fig = Figure(figsize=(7.4, 6.0), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frm_plot)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True, padx=6, pady=(6,0))
        toolbar = NavigationToolbar2Tk(self.canvas, self.frm_plot, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.update_plot()

    def _rebuild_bounds(self):
        nx, ny, nz = self.nx_var.get(), self.ny_var.get(), self.nz_var.get()
        self.bounds_cache = build_box_shell(nx, ny, nz, self.bound_thickness_var.get())
        self.update_plot()

    def add_tree(self):
        nx, ny, nz = self.nx_var.get(), self.ny_var.get(), self.nz_var.get()
        x, y, z0 = self.tx_var.get(), self.ty_var.get(), self.tz_var.get()
        r, h = self.tr_var.get(), self.th_var.get()
        if not (0 <= x < nx and 0 <= y < ny and 0 <= z0 < nz):
            messagebox.showerror("Out of bounds", "Tree base must lie inside the map.")
            return
        if r <= 0 or h <= 0:
            messagebox.showerror("Invalid size", "Radius and height must be positive.")
            return
        self.trees.append((x, y, z0, r, h))
        self.listbox.insert(tk.END, f"{len(self.trees):>3}: (x={x}, y={y}, z0={z0}, r={r}, h={h})")
        self.update_plot(redraw_only=True)

    def spawn_random(self):
        n = int(self.rand_n_var.get())
        if n <= 0:
            return
        nx, ny, nz = self.nx_var.get(), self.ny_var.get(), self.nz_var.get()
        z0, r, h = self.tz_var.get(), self.tr_var.get(), self.th_var.get()

        if r <= 0 or h <= 0:
            messagebox.showerror("Invalid size", "Radius and height must be positive.")
            return
        if not (0 <= z0 < nz):
            messagebox.showerror("Out of bounds", "Z0 must lie inside the map.")
            return

        # ensure cylinder footprint stays inside XY bounds
        x_min, x_max = r, max(0, nx - 1 - r)
        y_min, y_max = r, max(0, ny - 1 - r)
        if x_min > x_max or y_min > y_max:
            messagebox.showerror("Map too small", "Map too small for given radius.")
            return

        added = 0
        for _ in range(n):
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            self.trees.append((x, y, z0, r, h))
            added += 1
            self.listbox.insert(tk.END, f"{len(self.trees):>3}: (x={x}, y={y}, z0={z0}, r={r}, h={h})")

        self.status_var.set(f"Spawned {added} random trees.")
        self.update_plot(redraw_only=True)

    def remove_selected(self):
        sel = list(self.listbox.curselection())
        if not sel:
            messagebox.showinfo("Remove", "Select one or more trees to remove.")
            return
        for idx in reversed(sel):
            self.listbox.delete(idx)
            del self.trees[idx]
        self.listbox.delete(0, tk.END)
        for i, (x,y,z0,r,h) in enumerate(self.trees, start=1):
            self.listbox.insert(tk.END, f"{i:>3}: (x={x}, y={y}, z0={z0}, r={r}, h={h})")
        self.update_plot(redraw_only=True)

    def clear_trees(self):
        self.trees.clear()
        self.listbox.delete(0, tk.END)
        self.voxels.clear()
        self.status_var.set("Cleared trees. Rebuild to update voxels.")
        self.update_plot()

    def on_apply_size(self):
        # changing size invalidates caches
        self.voxels.clear()
        self.bounds_cache = set()
        self.status_var.set(f"Map size set to {self.nx_var.get()}×{self.ny_var.get()}×{self.nz_var.get()}.")
        self.update_plot()

    def on_build(self):
        nx, ny, nz = self.nx_var.get(), self.ny_var.get(), self.nz_var.get()
        self.voxels = build_forest_voxels(self.trees, nx, ny, nz)
        # update bounds cache (so preview/export use the latest thickness)
        self._rebuild_bounds()
        self.status_var.set(f"Built: {len(self.voxels)} tree voxels, {len(self.bounds_cache)} bound voxels (T={self.bound_thickness_var.get()}).")
        self.update_plot()

    def on_export(self):
        if not self.voxels and not self.include_bounds_export_var.get():
            messagebox.showwarning("Nothing to export", "Build the map first, or enable bounds export.")
            return
        # ensure bounds cache is current
        if not self.bounds_cache:
            self._rebuild_bounds()

        export_set = set(self.voxels)
        if self.include_bounds_export_var.get():
            export_set |= self.bounds_cache

        if not export_set:
            messagebox.showwarning("Nothing to export", "No voxels to export.")
            return

        path = filedialog.asksaveasfilename(
            title="Save occupied voxels",
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            vox_sorted = sorted(export_set)
            with open(path, "w", encoding="utf-8") as f:
                for x, y, z in vox_sorted:
                    f.write(f"{x} {y} {z}\n")
            messagebox.showinfo("Export complete",
                                f"Saved {len(vox_sorted)} voxels (trees{' + bounds' if self.include_bounds_export_var.get() else ''}).")
        except Exception as e:
            messagebox.showerror("Export failed", f"Could not save file:\n{e}")

    def _draw_map_box_wire(self, nx, ny, nz):
        xs = [0, nx]; ys = [0, ny]; zs = [0, nz]
        corners = [(xs[i], ys[j], zs[k]) for i in (0,1) for j in (0,1) for k in (0,1)]
        edges = [
            (0,1),(0,2),(0,4),
            (3,1),(3,2),(3,7),
            (5,1),(5,4),(5,7),
            (6,2),(6,4),(6,7),
        ]
        for a,b in edges:
            xa,ya,za = corners[a]
            xb,yb,zb = corners[b]
            self.ax.plot([xa,xb],[ya,yb],[za,zb], linewidth=0.8)

    def _set_equal_aspect(self, nx, ny, nz):
        self.ax.set_xlim(0, nx)
        self.ax.set_ylim(0, ny)
        self.ax.set_zlim(0, nz)
        try:
            self.ax.set_box_aspect([nx, ny, nz])
        except Exception:
            pass

    def update_plot(self, redraw_only=False):
        self.ax.clear()
        self.ax.set_xlabel("X"); self.ax.set_ylabel("Y"); self.ax.set_zlabel("Z")
        self.ax.set_title("Forest (cylinders)")

        nx, ny, nz = self.nx_var.get(), self.ny_var.get(), self.nz_var.get()
        self._draw_map_box_wire(nx, ny, nz)

        # Preview: union tree voxels + (optional) bounds voxels
        preview_set = set(self.voxels)
        if self.include_bounds_preview_var.get():
            if not self.bounds_cache:
                self._rebuild_bounds()
            preview_set |= self.bounds_cache

        if preview_set:
            vmax = max(2000, int(self.max_preview_var.get()))
            alpha = max(0.05, min(1.0, float(self.alpha_var.get())))
            pts = list(preview_set)
            if len(pts) > vmax:
                pts = random.sample(pts, k=vmax)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            self.ax.scatter(xs, ys, zs, s=float(self.tree_point_size_var.get()), alpha=alpha)

        # show individual tree centers as small markers at base
        for (x, y, z0, r, h) in self.trees:
            self.ax.scatter([x],[y],[z0], s=20)

        self._set_equal_aspect(nx, ny, nz)
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = TreeSpawnerApp()
    app.mainloop()
